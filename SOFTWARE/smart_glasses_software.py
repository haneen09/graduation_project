import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import uuid
import os
import time
import threading

# allowed classes
ALLOWED_CLASSES = {
    "person",
    "car",
    "bus",
    "bicycle",
    "traffic light",
    "chair",
    "couch",
    "dining table",
    "bed",
    "tv"
}

# speech control
last_speech_time = 0
SPEECH_COOLDOWN = 5
last_objects_spoken = set()

# load YOLO model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# load OCR
print("Loading OCR model...")
reader = easyocr.Reader(['en'])


def log_detection(text):
    with open("detections_log.txt", "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {text}\n")


def speak(text):
    if text.strip() == "":
        return

    print(f"Speaking: {text}")
    log_detection(text)

    def _speak():
        filename = f"speech_{uuid.uuid4()}.mp3"
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            os.startfile(filename)
            time.sleep(4)
        except Exception as e:
            print(f"Audio playback error: {e}")

    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()


def yolo8_detect(frame):
    results = model(
        frame,
        stream=True,
        verbose=False,
        conf=0.60
    )

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in ALLOWED_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append((label, conf, (x1, y1, x2, y2)))

    return detections


def draw_detections(frame, detections):
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return frame


def run_ocr(cropped_img):
    if cropped_img is None or cropped_img.size == 0:
        return ""

    results = reader.readtext(cropped_img)
    texts = []

    for (_, text, confidence) in results:
        if confidence > 0.4:
            texts.append(text)

    return " ".join(texts)


def run_ocr_full(frame):
    if frame is None or frame.size == 0:
        return ""

    results = reader.readtext(frame)
    texts = []

    for (_, text, confidence) in results:
        if confidence > 0.4:
            texts.append(text)

    return " ".join(texts)


def get_object_direction(box, frame_width):
    x1, _, x2, _ = box
    center_x = (x1 + x2) // 2

    if center_x < frame_width / 3:
        return "on your left"
    elif center_x < (2 * frame_width) / 3:
        return "in front of you"
    else:
        return "on your right"


# image mode
def process_image(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error reading image.")
        return

    detections = yolo8_detect(frame)
    detected_objects = []
    detected_texts = []

    frame_width = frame.shape[1]
    for label, conf, (x1, y1, x2, y2) in detections:
        direction = get_object_direction((x1, y1, x2, y2), frame_width)
        detected_objects.append(f"{label} {direction}")
        crop = frame[y1:y2, x1:x2]
        text = run_ocr(crop)
        if text != "":
            detected_texts.append(text)

    # add this line right here
    full_text = run_ocr_full(frame)
    if full_text != "":
        detected_texts.append(full_text)

    output = draw_detections(frame.copy(), detections)
    cv2.imshow("Detection Result", output)

    speech_parts = []

    if detected_objects:
        unique_objects = list(set(detected_objects))
        speech_parts.append("I see " + ", ".join(unique_objects))

    if detected_texts:
        unique_texts = list(set(detected_texts))
        speech_parts.append("Text detected: " + ", ".join(unique_texts))

    final_sentence = ". ".join(speech_parts)

    if final_sentence != "":
        print(final_sentence)
        speak(final_sentence)
    else:
        print("No allowed objects detected.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# camera mode
def run_camera_mode():
    global last_speech_time
    global last_objects_spoken

    print("\nStarting camera mode...")

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))
        detections = yolo8_detect(frame)
        output = draw_detections(frame.copy(), detections)
        cv2.imshow("Smart Glasses Vision", output)

        detected_objects = []
        frame_width = frame.shape[1]

        for label, conf, box in detections:
            direction = get_object_direction(box, frame_width)
            detected_objects.append(f"{label} {direction}")

        frame_count += 1

        if detected_objects and frame_count % 20 == 0:
            unique_objects = set(detected_objects)
            current_time = time.time()

            if (
                unique_objects != last_objects_spoken
                and current_time - last_speech_time > SPEECH_COOLDOWN
            ):
                description = ", ".join(unique_objects)
                speak(description)
                last_speech_time = current_time
                last_objects_spoken = unique_objects

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        elif key == ord('o'):
            detected_text = run_ocr_full(frame)
            if detected_text != "":
                print(f"Text found: {detected_text}")
                speak(f"Text says {detected_text}")
            else:
                print("No text found.")

    cap.release()
    cv2.destroyAllWindows()


# main
if __name__ == "__main__":
    print("\nSMART GLASSES SYSTEM")
    print("1 - Image Mode")
    print("2 - Camera Mode")

    choice = input("Select mode: ")

    if choice == "1":
        image_path = input("Enter image path: ")
        process_image(image_path)

    elif choice == "2":
        run_camera_mode()

    else:
        print("Invalid choice.")