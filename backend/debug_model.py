import os
import cv2
import glob
from ultralytics import YOLO
import torch

# Patch torch
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim

# Find latest file
upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
files = glob.glob(os.path.join(upload_dir, "*"))
if not files:
    print("No files found.")
    exit()
    
latest_file = max(files, key=os.path.getctime)
print(f"analyzing: {latest_file}")

# Load OIV7
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov8s-oiv7.pt"))
model = YOLO(model_path)
names = model.names

# interesting = ["ladder", "scaffold", "mewp", "platform", "lift", "crane", "bucket"]
# print("--- SEARCHING CLASSES ---")
# for id, name in names.items():
#     if any(i in name.lower() for i in interesting):
#         print(f"{id}: {name}")
# exit()

results = model.predict(cv2.imread(latest_file), conf=0.01, iou=0.45, verbose=False)

with open("detections_log.txt", "w") as f:
    for box in results[0].boxes:
       cls_id = int(box.cls[0])
       name = results[0].names[cls_id]
       conf = float(box.conf[0])
       f.write(f"Class: {name}, Conf: {conf:.3f}\n")
print("Detections written to detections_log.txt")

print("\n--- DETECTIONS ---")
for box in results[0].boxes:
   cls_id = int(box.cls[0])
   name = results[0].names[cls_id]
   conf = float(box.conf[0])
   print(f"Class: {name}, Conf: {conf:.3f}")




print("\n--- DETECTIONS ---")
seen_classes = set()
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    name = results[0].names[cls_id]
    conf = float(box.conf[0])
# print(f"Class: {name}, Conf: {conf:.3f}")
    seen_classes.add(name)

print("\n--- UNIQUE CLASSES SEEN ---")
print(sorted(list(seen_classes)))

