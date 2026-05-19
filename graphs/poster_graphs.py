import matplotlib.pyplot as plt

# -----------------------------
# Graph 1: Object Detection Frequency
# -----------------------------

objects = ["Person", "Car", "Chair", "Bicycle", "Traffic Light", "Couch", "TV", "Dining Table"]
counts = [8, 6, 6, 5, 4, 3, 3, 2]

plt.figure()

plt.bar(objects, counts, color="#C8A2C8")  # light purple

plt.title("Object Detection Frequency")
plt.xlabel("Object Class")
plt.ylabel("Number of Detections")

plt.xticks(rotation=45)

plt.savefig("object_detection_frequency.png", dpi=300, bbox_inches="tight")

plt.close()


# -----------------------------
# Graph 2: Detection Accuracy
# -----------------------------

labels = ["Correct Detections", "Incorrect Detections"]
values = [43, 10]

colors = ["#6a329f", "#C8A2C8"]  # dark purple + light purple

plt.figure()

plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)

plt.title("Detection Accuracy")

plt.savefig("detection_accuracy.png", dpi=300, bbox_inches="tight")

plt.close()


# -----------------------------
# Graph 3: Processing Time
# -----------------------------

modules = ["YOLOv8 Detection", "OCR Processing", "Text-to-Speech"]
times = [1.0, 0.5, 0.3]

plt.figure()

plt.bar(modules, times, color="#C8A2C8")  # light purple

plt.title("Average Processing Time per Module")
plt.xlabel("System Module")
plt.ylabel("Processing Time (seconds)")

plt.xticks(rotation=20)

plt.savefig("processing_time_modules.png", dpi=300, bbox_inches="tight")

plt.close()

print("Graphs generated successfully!")
