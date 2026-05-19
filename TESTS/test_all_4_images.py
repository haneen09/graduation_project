from smart_glasses_windows import process_image, process_ocr_image

# Clear previous results
open("detection_results.txt", "w").close()

print("\n" + "="*70)
print("TESTING ALL 6 IMAGES - SMART GLASSES VISION SYSTEM")
print("="*70)

print("\nOUTDOOR TEST 1")
print("="*70)
process_image('outdoor1.jpg', "detection_results.txt")

print("\nOUTDOOR TEST 2")
print("="*70)
process_image('outdoor2.jpg', "detection_results.txt")

print("\nINDOOR TEST 1")
print("="*70)
process_image('indoor1.jpg', "detection_results.txt")

print("\nINDOOR TEST 2")
print("="*70)
process_image('indoor2.jpg', "detection_results.txt")

print("\nOCR TEST 1 - Text Image")
print("="*70)
process_ocr_image('ocr1.png', "detection_results.txt")

print("\nOCR TEST 2 - Text Image")
print("="*70)
process_ocr_image('ocr2.png', "detection_results.txt")

print("\n" + "="*70)
print("ALL TESTS COMPLETE!")
print("Results saved to detection_results.txt")
print("="*70 + "\n")