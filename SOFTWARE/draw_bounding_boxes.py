import cv2
from ultralytics import YOLO
import easyocr
import os

# ==============================
# LOAD YOLO MODEL
# ==============================
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# ==============================
# LOAD OCR READER
# ==============================
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])

# ==============================
# ALLOWED YOLO CLASSES
# ==============================
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

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
YOLO_OUTPUT = "bounding_box_results"
OCR_OUTPUT = "ocr_results"

os.makedirs(YOLO_OUTPUT, exist_ok=True)
os.makedirs(OCR_OUTPUT, exist_ok=True)

# =========================================================
# YOLO OBJECT DETECTION FUNCTION
# =========================================================
def draw_bounding_boxes(image_path):
    
    print(f"\nProcessing YOLO image: {image_path}")

    frame = cv2.imread(image_path)

    if frame is None:
        print("Error reading image")
        return

    results = model(frame)

    detection_count = 0
    class_counts = {}

    for r in results:

        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            # ONLY KEEP YOUR TARGET CLASSES
            if label not in ALLOWED_CLASSES:
                continue

            detection_count += 1

            if label not in class_counts:
                class_counts[label] = 0

            class_counts[label] += 1

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # LABEL TEXT
            text = f"{label} {conf:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # SAVE OUTPUT IMAGE
    output_path = os.path.join(
        YOLO_OUTPUT,
        f"bbox_{os.path.basename(image_path)}"
    )

    cv2.imwrite(output_path, frame)

    print(f"Detections: {detection_count}")
    print(f"Saved to: {output_path}")

    return {
        "image": image_path,
        "output": output_path,
        "detections": detection_count,
        "classes": class_counts
    }

# =========================================================
# OCR FUNCTION
# =========================================================
def run_ocr(image_path):

    print(f"\nProcessing OCR image: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print("Error reading OCR image")
        return

    # ==========================================
    # PREPROCESSING
    # ==========================================

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding for cleaner text
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # ==========================================
    # OCR
    # ==========================================

    results = reader.readtext(thresh)

    detected_texts = []

    for detection in results:

        bbox, text, confidence = detection

        # Ignore weak detections
        if confidence < 0.4:
            continue

        detected_texts.append(text)

        # BOX COORDINATES
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # DRAW BOX
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # DRAW TEXT
        cv2.putText(
            image,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # SAVE RESULT
    output_path = os.path.join(
        OCR_OUTPUT,
        f"ocr_{os.path.basename(image_path)}"
    )

    cv2.imwrite(output_path, image)

    print(f"Detected Text: {detected_texts}")
    print(f"Saved to: {output_path}")

    return {
        "image": image_path,
        "output": output_path,
        "texts": detected_texts
    }

# =========================================================
# TEST IMAGES
# =========================================================

# OBJECT DETECTION IMAGES
yolo_images = [
    'outdoor1.jpg',
    'outdoor2.jpg',
    'indoor1.jpg',
    'indoor2.jpg'
]

# OCR IMAGES
ocr_images = [
    'ocr1.png',
    'ocr2.png'
]

# =========================================================
# RUN YOLO TESTING
# =========================================================
print("\n" + "="*70)
print("YOLO OBJECT DETECTION")
print("="*70)

yolo_results = []

for image in yolo_images:

    if os.path.exists(image):

        result = draw_bounding_boxes(image)

        if result:
            yolo_results.append(result)

    else:
        print(f"{image} not found")

# =========================================================
# RUN OCR TESTING
# =========================================================
print("\n" + "="*70)
print("OCR TEXT DETECTION")
print("="*70)

ocr_results = []

for image in ocr_images:

    if os.path.exists(image):

        result = run_ocr(image)

        if result:
            ocr_results.append(result)

    else:
        print(f"{image} not found")

# =========================================================
# FINAL SUMMARY REPORT
# =========================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

with open("final_report.txt", "w") as f:

    f.write("AI SMART VISION SYSTEM REPORT\n")
    f.write("="*70 + "\n\n")

    # YOLO RESULTS
    f.write("YOLO OBJECT DETECTION RESULTS\n")
    f.write("="*70 + "\n")

    total_detections = 0

    for result in yolo_results:

        f.write(f"\nImage: {result['image']}\n")
        f.write(f"Detections: {result['detections']}\n")

        for cls, count in result['classes'].items():
            f.write(f"  - {cls}: {count}\n")

        total_detections += result['detections']

    f.write(f"\nTOTAL OBJECT DETECTIONS: {total_detections}\n\n")

    # OCR RESULTS
    f.write("OCR RESULTS\n")
    f.write("="*70 + "\n")

    total_texts = 0

    for result in ocr_results:

        f.write(f"\nImage: {result['image']}\n")

        for text in result['texts']:
            f.write(f"  - {text}\n")
            total_texts += 1

    f.write(f"\nTOTAL TEXTS DETECTED: {total_texts}\n")

print("\nResults saved successfully.")
print(f"YOLO images folder: {YOLO_OUTPUT}")
print(f"OCR images folder: {OCR_OUTPUT}")
print("Report file: final_report.txt")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)