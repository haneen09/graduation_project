from smart_glasses_rpi_sim import process_image_with_obstacle_detection, process_ocr_image_with_obstacle

# Clear previous results
open("rpi_simulation_results.txt", "w").close()

print("\n" + "="*70)
print("RASPBERRY PI HARDWARE SIMULATION - ALL TESTS")
print("="*70)

print("\nOUTDOOR TEST 1 - Object Detection + Obstacle Detection")
print("="*70)
process_image_with_obstacle_detection('outdoor1.jpg', "rpi_simulation_results.txt")

print("\nOUTDOOR TEST 2 - Object Detection + Obstacle Detection")
print("="*70)
process_image_with_obstacle_detection('outdoor2.jpg', "rpi_simulation_results.txt")

print("\nINDOOR TEST 1 - Object Detection + Obstacle Detection")
print("="*70)
process_image_with_obstacle_detection('indoor1.jpg', "rpi_simulation_results.txt")

print("\nINDOOR TEST 2 - Object Detection + Obstacle Detection")
print("="*70)
process_image_with_obstacle_detection('indoor2.jpg', "rpi_simulation_results.txt")

print("\nOCR TEST 1 - Text Recognition + Obstacle Detection")
print("="*70)
process_ocr_image_with_obstacle('ocr1.png', "rpi_simulation_results.txt")

print("\nOCR TEST 2 - Text Recognition + Obstacle Detection")
print("="*70)
process_ocr_image_with_obstacle('ocr2.png', "rpi_simulation_results.txt")

print("\n" + "="*70)
print("ALL HARDWARE SIMULATION TESTS COMPLETE!")
print("Results saved to rpi_simulation_results.txt")
print("="*70 + "\n")