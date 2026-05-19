import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pyttsx3
from pathlib import Path

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# Initialize OCR reader
print("Loading OCR model...")
reader = easyocr.Reader(['en'])

def speak(text):
    """Convert text to speech"""
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def process_image(image_path):
    """Process an image and detect objects + text"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # YOLOv8 detection
    print("Running object detection...")
    results = model(image)
    
    # Extract detections
    detections = results[0].boxes
    detected_objects = []
    
    for box in detections:
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        class_name = model.names[class_id]
        detected_objects.append(f"{class_name} with {confidence:.2f} confidence")
    
    # Text detection with OCR
    print("Running OCR...")
    ocr_results = reader.readtext(image)
    detected_text = [text[1] for text in ocr_results]
    
    # Create summary
    summary = "Scene analysis: "
    if detected_objects:
        summary += f"I detected {', '.join(detected_objects[:3])}. "
    if detected_text:
        summary += f"Text found: {' '.join(detected_text[:2])}"
    
    print(f"\n{summary}\n")
    speak(summary)

# Test with a sample image
if __name__ == "__main__":
    print("Smart Glasses Vision System - Test Mode")
    print("=" * 50)
    
    # For now, just test that everything loads
    print("✓ YOLOv8 loaded")
    print("✓ OCR loaded")
    print("✓ TTS ready")
    print("\nSystem ready! To test with images, use: process_image('your_image.jpg')")