# Smart Glasses Vision System 

An AI powered assistive vision system built for visually impaired users.
Detects objects, reads text (OCR), and speaks descriptions out loud in real time.

## Features
- Real-time object detection using YOLOv8
- OCR text reading using EasyOCR
- Text-to-speech audio output using gTTS
- Direction detection (left, center, right)
- Image mode and live camera mode
- Detection logging with timestamps

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- EasyOCR
- OpenCV
- gTTS

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python SOFTWARE/smart_glasses_software.py`
3. Select mode 1 (image) or 2 (camera)
