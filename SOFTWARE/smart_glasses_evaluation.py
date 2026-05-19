import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import uuid
import os
import time
import random
import json

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO not available - running in simulation mode")
    GPIO = None

# simulation mode flag
SIMULATION_MODE = GPIO is None

# 10 target classes selected from COCO dataset for this project
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

# GPIO pins for HC-SR04 ultrasonic sensor
# GPIO 23 = physical pin 16 (Trig), GPIO 24 = physical pin 18 (Echo)
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24

if not SIMULATION_MODE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
    GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)

# load YOLOv8n model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# load EasyOCR reader
print("Loading OCR model...")
reader = easyocr.Reader(['en'])

# metrics storage for evaluation summary
metrics = {
    "total_images": 0,
    "total_objects_detected": 0,
    "average_detection_confidence": 0,
    "detection_confidences": [],
    "ocr_texts_found": 0,
    "average_ocr_confidence": 0,
    "ocr_confidences": [],
    "obstacle_detections": 0,
    "processing_times": [],
    "class_accuracy": {}
}

def measure_distance():
    # return simulated distance if running without hardware
    if SIMULATION_MODE:
        distance = random.uniform(20, 200)
        return distance
    else:
        # trigger ultrasonic pulse
        GPIO.output(ULTRASONIC_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(ULTRASONIC_TRIG, False)
        
        # record pulse start time
        pulse_start = time.time()
        while GPIO.input(ULTRASONIC_ECHO) == 0:
            pulse_start = time.time()
        
        # record pulse end time
        pulse_end = time.time()
        while GPIO.input(ULTRASONIC_ECHO) == 1:
            pulse_end = time.time()
        
        # calculate distance in centimeters
        distance = (pulse_end - pulse_start) * 17150
        return distance

def trigger_obstacle_warning(distance):
    # check if obstacle is within 30cm threshold
    if distance < 30:
        return True
    return False

def yolo8_detect(frame):
    # run YOLOv8 detection and filter to target classes only
    results = model(frame, stream=True)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label not in ALLOWED_CLASSES:
                continue
            
            detections.append((label, conf, (x1, y1, x2, y2)))
    
    return detections

def draw_detections(frame, detections):
    # draw bounding boxes and class labels on frame
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def run_ocr(cropped_img):
    # run OCR on cropped image region
    if cropped_img is None or cropped_img.size == 0:
        return [], []
    
    results = reader.readtext(cropped_img)
    texts = []
    confidences = []
    for (_, text, confidence) in results:
        if confidence > 0.4:
            texts.append(text)
            confidences.append(confidence)
    
    return texts, confidences

def run_ocr_full(frame):
    # run OCR on full image frame
    if frame is None or frame.size == 0:
        return [], []
    
    results = reader.readtext(frame)
    texts = []
    confidences = []
    for (_, text, confidence) in results:
        if confidence > 0.4:
            texts.append(text)
            confidences.append(confidence)
    
    return texts, confidences

def process_image_with_metrics(image_path, log_file="evaluation_results.txt"):
    # process image and collect evaluation metrics
    start_time = time.time()
    
    print(f"\nProcessing image: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # measure obstacle distance
    distance = measure_distance()
    obstacle_detected = trigger_obstacle_warning(distance)
    
    # run object detection
    detections = yolo8_detect(frame)
    
    detected_objects = []
    detected_confidences = []
    detected_texts = []
    detected_text_confidences = []
    
    # process each detection
    for label, conf, (x1, y1, x2, y2) in detections:
        detected_objects.append(label)
        detected_confidences.append(conf)
        metrics["detection_confidences"].append(conf)
        
        print(f"  Detected: {label} (Confidence: {conf*100:.1f}%)")
        
        # run OCR on detected object region
        crop = frame[y1:y2, x1:x2]
        texts, ocr_confs = run_ocr(crop)
        
        if texts:
            for text, ocr_conf in zip(texts, ocr_confs):
                detected_texts.append(text)
                detected_text_confidences.append(ocr_conf)
                metrics["ocr_confidences"].append(ocr_conf)
                print(f"    OCR: {text} (Confidence: {ocr_conf*100:.1f}%)")
    
    # draw bounding boxes for visualization
    frame = draw_detections(frame, detections)
    
    processing_time = time.time() - start_time
    metrics["processing_times"].append(processing_time)
    
    # update metrics
    metrics["total_images"] += 1
    metrics["total_objects_detected"] += len(detected_objects)
    metrics["ocr_texts_found"] += len(detected_texts)
    if obstacle_detected:
        metrics["obstacle_detections"] += 1
    
    # update per-class accuracy statistics
    for obj in detected_objects:
        if obj not in metrics["class_accuracy"]:
            metrics["class_accuracy"][obj] = {"count": 0, "total_confidence": 0}
        metrics["class_accuracy"][obj]["count"] += 1
        if detected_confidences:
            metrics["class_accuracy"][obj]["total_confidence"] += detected_confidences[detected_objects.index(obj)]
    
    # write results to log file
    with open(log_file, "a") as f:
        f.write(f"\nImage: {image_path}\n")
        f.write(f"Processing Time: {processing_time:.2f}s\n")
        f.write(f"Distance (cm): {distance:.1f}\n")
        f.write(f"Obstacle Detected: {obstacle_detected}\n")
        f.write(f"\nObject Detection Results:\n")
        f.write(f"  Total Objects: {len(detected_objects)}\n")
        if detected_confidences:
            avg_detection_conf = np.mean(detected_confidences) * 100
            f.write(f"  Average Detection Confidence: {avg_detection_conf:.1f}%\n")
            f.write(f"  Min Confidence: {min(detected_confidences)*100:.1f}%\n")
            f.write(f"  Max Confidence: {max(detected_confidences)*100:.1f}%\n")
        f.write(f"  Detected Objects: {', '.join(detected_objects)}\n")
        f.write(f"\nOCR Results:\n")
        f.write(f"  Total Text Found: {len(detected_texts)}\n")
        if detected_text_confidences:
            avg_ocr_conf = np.mean(detected_text_confidences) * 100
            f.write(f"  Average OCR Confidence: {avg_ocr_conf:.1f}%\n")
            f.write(f"  Min OCR Confidence: {min(detected_text_confidences)*100:.1f}%\n")
            f.write(f"  Max OCR Confidence: {max(detected_text_confidences)*100:.1f}%\n")
        f.write(f"  Detected Text: {', '.join(detected_texts)}\n")
        f.write("-" * 60 + "\n")

def process_ocr_image_with_metrics(image_path, log_file="evaluation_results.txt"):
    # process text-only image with metrics
    start_time = time.time()
    
    print(f"\nProcessing OCR image: {image_path}")
    
    distance = measure_distance()
    obstacle_detected = trigger_obstacle_warning(distance)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    texts, ocr_confs = run_ocr_full(frame)
    
    processing_time = time.time() - start_time
    metrics["processing_times"].append(processing_time)
    
    metrics["total_images"] += 1
    metrics["ocr_texts_found"] += len(texts)
    if obstacle_detected:
        metrics["obstacle_detections"] += 1
    
    for conf in ocr_confs:
        metrics["ocr_confidences"].append(conf)
    
    with open(log_file, "a") as f:
        f.write(f"\nOCR Image: {image_path}\n")
        f.write(f"Processing Time: {processing_time:.2f}s\n")
        f.write(f"Distance (cm): {distance:.1f}\n")
        f.write(f"Obstacle Detected: {obstacle_detected}\n")
        f.write(f"\nOCR Results:\n")
        f.write(f"  Total Text Found: {len(texts)}\n")
        if ocr_confs:
            avg_ocr_conf = np.mean(ocr_confs) * 100
            f.write(f"  Average OCR Confidence: {avg_ocr_conf:.1f}%\n")
            f.write(f"  Min OCR Confidence: {min(ocr_confs)*100:.1f}%\n")
            f.write(f"  Max OCR Confidence: {max(ocr_confs)*100:.1f}%\n")
        f.write(f"  Detected Text: {', '.join(texts)}\n")
        f.write("-" * 60 + "\n")
    
    if texts:
        print(f"Text found: {', '.join(texts)}")
        for text, conf in zip(texts, ocr_confs):
            print(f"  - {text} (Confidence: {conf*100:.1f}%)")
    else:
        print("No text found in image.")

def save_metrics_summary(log_file="evaluation_results.txt"):
    # save overall evaluation summary to log
    with open(log_file, "a") as f:
        f.write("\n\n" + "="*60 + "\n")
        f.write("OVERALL EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Total Images Processed: {metrics['total_images']}\n")
        f.write(f"Total Objects Detected: {metrics['total_objects_detected']}\n")
        
        if metrics['detection_confidences']:
            avg_detection = np.mean(metrics['detection_confidences']) * 100
            f.write(f"Average Detection Confidence: {avg_detection:.1f}%\n")
        
        f.write(f"Total Text Found: {metrics['ocr_texts_found']}\n")
        
        if metrics['ocr_confidences']:
            avg_ocr = np.mean(metrics['ocr_confidences']) * 100
            f.write(f"Average OCR Confidence: {avg_ocr:.1f}%\n")
        
        f.write(f"Obstacle Detections: {metrics['obstacle_detections']}\n")
        
        if metrics['processing_times']:
            avg_time = np.mean(metrics['processing_times'])
            f.write(f"Average Processing Time: {avg_time:.2f}s\n")
        
        f.write(f"\nPer-Class Detection Accuracy:\n")
        for class_name, class_data in metrics['class_accuracy'].items():
            avg_conf = (class_data['total_confidence'] / class_data['count'] * 100) if class_data['count'] > 0 else 0
            f.write(f"  {class_name}: {class_data['count']} detections, {avg_conf:.1f}% avg confidence\n")

if __name__ == "__main__":
    print("SMART GLASSES - EVALUATION METRICS")
    print("="*60)
    print(f"\nMode: {'SIMULATION' if SIMULATION_MODE else 'HARDWARE'}")
    print("YOLOv8 loaded")
    print("OCR loaded")
    print("Ultrasonic Sensor: Simulated")
    print("\nFiltering for 10 classes:")
    for cls in sorted(ALLOWED_CLASSES):
        print(f"   - {cls}")
