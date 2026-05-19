import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import uuid
import os
import time
import random

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO not available - running in simulation mode")
    GPIO = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
    print("picamera2 not available")

# ==========================
# SIMULATION MODE
# ==========================
SIMULATION_MODE = GPIO is None

# ==========================
# ALLOWED CLASSES
# ==========================
ALLOWED_CLASSES = {
    "person", "car", "bus", "bicycle", "traffic light",
    "chair", "couch", "dining table", "bed", "tv"
}

# ==========================
# SPEECH CONTROL
# ==========================
last_object_speech_time = 0
last_obstacle_speech_time = 0
OBJECT_COOLDOWN = 5
OBSTACLE_COOLDOWN = 3
last_objects_spoken = set()

# ==========================
# OBSTACLE DISTANCE ZONES (cm)
# ==========================
VERY_CLOSE = 30
CLOSE = 60
MEDIUM = 100

# ==========================
# GPIO SETUP — CORRECTED FOR YOUR WIRING
# ==========================
ULTRASONIC_TRIG = 23   # Red wire    → Pin 16
ULTRASONIC_ECHO = 24   # Orange wire → Pin 18 (through resistor divider)

if not SIMULATION_MODE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
    GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)

# ==========================
# SYSTEM ALWAYS ACTIVE (no button)
# ==========================
system_active = True

# ==========================
# LOAD MODELS
# ==========================
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

print("Loading OCR model...")
reader = easyocr.Reader(['en'])


# ==========================
# SPEECH FUNCTION
# ==========================
def speak(text):
    print(f"Speaking: {text}")

    # folder for saved audio
    save_folder = "/home/haneen/shared_audio"

    # create folder if missing
    os.makedirs(save_folder, exist_ok=True)

    filename = f"{save_folder}/latest.mp3"

    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        print(f"Saved audio to: {filename}")

    except Exception as e:
        print(f"Audio error: {e}")


# ==========================
# ULTRASONIC DISTANCE
# ==========================
def measure_distance():
    if SIMULATION_MODE:
        distance = random.uniform(20, 200)
        print(f"[SIMULATED] Distance: {distance:.1f} cm")
        return distance

    GPIO.output(ULTRASONIC_TRIG, False)
    time.sleep(0.1)  # let sensor settle

    GPIO.output(ULTRASONIC_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()

    timeout = time.time() + 0.04
    while GPIO.input(ULTRASONIC_ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            print("Sensor timeout (no pulse start)")
            return 999

    timeout = time.time() + 0.04
    while GPIO.input(ULTRASONIC_ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            print("Sensor timeout (no pulse end)")
            return 999

    distance = (pulse_end - pulse_start) * 17150
    return round(distance, 2)


# ==========================
# OBSTACLE WARNING
# ==========================
def trigger_obstacle_warning(distance):
    global last_obstacle_speech_time
    current_time = time.time()

    if current_time - last_obstacle_speech_time < OBSTACLE_COOLDOWN:
        return False

    if distance < VERY_CLOSE:
        speak(f"Stop. Obstacle very close at {int(distance)} centimeters.")
    elif distance < CLOSE:
        speak(f"Obstacle nearby at {int(distance)} centimeters.")
    elif distance < MEDIUM:
        speak(f"Object ahead at {int(distance)} centimeters.")
    else:
        return False

    last_obstacle_speech_time = current_time
    return True


# ==========================
# YOLO DETECTION
# ==========================
def yolo8_detect(frame):
    results = model(frame, stream=True, verbose=False, conf=0.60)  # filter at model level too
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label not in ALLOWED_CLASSES:  # check class FIRST before anything else
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append((label, conf, (x1, y1, x2, y2)))
    return detections


def draw_detections(frame, detections):
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def get_object_direction(box, frame_width):
    x1, _, x2, _ = box
    center_x = (x1 + x2) // 2
    if center_x < frame_width / 3:
        return "on your left"
    elif center_x < (2 * frame_width) / 3:
        return "in front of you"
    else:
        return "on your right"


# ==========================
# OCR FUNCTIONS
# ==========================
def run_ocr(cropped_img):
    if cropped_img is None or cropped_img.size == 0:
        return ""
    results = reader.readtext(cropped_img)
    return " ".join(text for (_, text, conf) in results if conf > 0.4)


def run_ocr_full(frame):
    if frame is None or frame.size == 0:
        return ""
    results = reader.readtext(frame)
    return " ".join(text for (_, text, conf) in results if conf > 0.4)


# ==========================
# IMAGE MODE
# ==========================
def process_image_with_obstacle_detection(image_path):
    global last_object_speech_time, last_objects_spoken

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error reading image.")
        return

    distance = measure_distance()
    trigger_obstacle_warning(distance)

    detections = yolo8_detect(frame)
    detected_objects = [label for label, _, _ in detections]

    if detected_objects:
        unique_objects = set(detected_objects)
        if unique_objects != last_objects_spoken:
            current_time = time.time()
            if current_time - last_object_speech_time > OBJECT_COOLDOWN:
                object_list = ", ".join(unique_objects)
                speak(f"I see {object_list}. Distance is {int(distance)} centimeters.")
                last_object_speech_time = current_time
                last_objects_spoken = unique_objects


# ==========================
# CAMERA MODE — Using picamera2 (libcamera)
# ==========================
def run_camera_mode():
    global last_object_speech_time, last_objects_spoken

    print("Starting camera mode...")
    
    picam2 = None
    cap = None
    
    # Try picamera2 first (new Pi OS)
    if Picamera2 is not None:
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            picam2.configure(config)
            picam2.start()
            print("Camera started with picamera2!")
        except Exception as e:
            print(f"picamera2 failed: {e}")
            picam2 = None
    
    # Fallback to OpenCV V4L2
    if picam2 is None:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Error: Could not access camera with any method")
            return
        print("Camera started with OpenCV V4L2")

    frame_count = 0
    try:
        while True:
            if picam2:
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))

            distance = measure_distance()
            trigger_obstacle_warning(distance)

            detections = yolo8_detect(frame)
            detected_objects = [label for label, _, _ in detections]

            if detected_objects and frame_count % 10 == 0:
                unique_objects = set(detected_objects)
                if unique_objects != last_objects_spoken:
                    current_time = time.time()
                    if current_time - last_object_speech_time > OBJECT_COOLDOWN:
                        frame_width = frame.shape[1]
                        spoken_descriptions = []
                        for label, _, box in detections:
                            direction = get_object_direction(box, frame_width)
                            spoken_descriptions.append(f"{label} {direction}")
                        description = ", ".join(set(spoken_descriptions))
                        speak(f"{description}. Distance is {int(distance)} centimeters.")
                        last_object_speech_time = current_time
                        last_objects_spoken = unique_objects

            # NO cv2.imshow — runs headless over SSH
            # Press Ctrl+C in terminal to stop

    except KeyboardInterrupt:
        print("\nStopping camera mode...")
    finally:
        if picam2:
            picam2.stop()
            picam2.close()
        if cap:
            cap.release()
        cleanup()


# ==========================
# CLEANUP
# ==========================
def cleanup():
    if not SIMULATION_MODE:
        GPIO.cleanup()


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("\nSMART GLASSES SYSTEM — ACTIVE")
    print("1 - Image Mode")
    print("2 - Camera Mode")

    choice = input("Select mode: ")

    if choice == "1":
        image_path = input("Enter image path: ")
        process_image_with_obstacle_detection(image_path)
    elif choice == "2":
        run_camera_mode()
    else:
        print("Invalid choice.")