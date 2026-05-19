from smart_glasses_evaluation import (
    process_image_with_metrics, 
    process_ocr_image_with_metrics,
    save_metrics_summary
)

# Clear previous results
open("evaluation_results.txt", "w").close()

print("\n" + "="*70)
print("SMART GLASSES - EVALUATION & METRICS COLLECTION")
print("="*70)

print("\nOUTDOOR TEST 1")
print("="*70)
process_image_with_metrics('outdoor1.jpg', "evaluation_results.txt")

print("\nOUTDOOR TEST 2")
print("="*70)
process_image_with_metrics('outdoor2.jpg', "evaluation_results.txt")

print("\nINDOOR TEST 1")
print("="*70)
process_image_with_metrics('indoor1.jpg', "evaluation_results.txt")

print("\nINDOOR TEST 2")
print("="*70)
process_image_with_metrics('indoor2.jpg', "evaluation_results.txt")

print("\nOCR TEST 1")
print("="*70)
process_ocr_image_with_metrics('ocr1.png', "evaluation_results.txt")

print("\nOCR TEST 2")
print("="*70)
process_ocr_image_with_metrics('ocr2.png', "evaluation_results.txt")

# Save summary
save_metrics_summary("evaluation_results.txt")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("Results saved to evaluation_results.txt")
print("="*70 + "\n")
