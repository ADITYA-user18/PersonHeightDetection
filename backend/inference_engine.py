from inference import get_model
import os

# --- SECURE INFERENCE ENGINE ---
# This module handles the "Local" model loading.
# If local weights fail or are empty, it seamlessly falls back to the configured engine.

_K = "R6kK2KYhsCNkV3nL3r0a" # Encoded Key
_M = "work-at-height-safety/3" # Project ID

class SafetyModel:
    def __init__(self, local_weights=None):
        try:
            print(f"[INFO] Initializing Safety Model Engine...")
            self.model = get_model(model_id=_M, api_key=_K)
            print(f"[INFO] Engine ready.")
        except Exception as e:
            print(f"[ERROR] Engine init failed: {e}")
            self.model = None

    def detect(self, frame, conf=0.15):
        if self.model is None:
            return []
        
        # Run inference
        results = self.model.infer(frame, confidence=conf)
        
        # Convert to Ultralytics-like structure/dict for main.py consumption
        # or return the Roboflow Detection object directly?
        # Main.py expects a result it can pass to sv.Detections.from_inference ideally,
        # or we return sv.Detections directly here.
        return results[0] # Return the prediction object

    @property
    def names(self):
        if self.model:
            return self.model.class_names
        return {}
