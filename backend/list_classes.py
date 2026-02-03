import torch
from ultralytics import YOLO
import os

# Patch torch
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim

try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov8s-oiv7.pt"))
    if not os.path.exists(model_path):
        model = YOLO('yolov8s-oiv7.pt')
    else:
        model = YOLO(model_path)
    
    with open("backend/oiv7_classes.txt", "w") as f:
        for id, name in model.names.items():
            f.write(f"{id}: {name}\n")
    print("Classes written to backend/oiv7_classes.txt")

except Exception as e:
    print(e)
