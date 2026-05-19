from ultralytics import YOLO

print("Downloading YOLOv8 model... This may take a minute")
model = YOLO('yolov8n.pt')
print("✓ Model downloaded successfully!")