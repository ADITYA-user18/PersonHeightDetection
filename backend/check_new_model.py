import torch
from ultralyticsplus import YOLO, render_result

# 1. Monkey patch torch.load
_original_load = torch.load

def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = safe_load_shim

try:
    print("Attempting to load keremberke/yolov8m-protective-equipment-detection...")
    model = YOLO('keremberke/yolov8m-protective-equipment-detection')
    print("Model loaded.")
    for id, name in model.model.names.items():
        print(f"{id}: {name}")

except Exception as e:
    print(f"Error: {e}")


