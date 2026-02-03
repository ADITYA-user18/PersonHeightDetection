from ultralytics import YOLO
import torch

# Workaround for torch 2.6
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim

try:
    model = YOLO("best (2).pt")
    print("Model loaded successfully.")
    print("Class Names:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
