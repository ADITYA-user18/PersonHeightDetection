import cv2
import numpy as np
import supervision as sv
from inference_engine import SafetyModel
import os
import time

# --- CONFIGURATION ---
model = SafetyModel()
HEIGHT_EQUIPMENT = ["ladder", "scaffold", "mewp", "scaffolding"]

def main():
    # 1. Get Input Video
    video_path = input("Enter the path to your video file: ").strip().strip('"').strip("'")
    if not os.path.exists(video_path):
        print(f"Error: File not found at '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loading and Processing {total_frames} frames into RAM...")
    print("Please wait... this ensures smooth playback later.")
    print("Using High-Accuracy Engine.")

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Store processed frames in memory (RAM)
    processed_frames_buffer = []
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing Frame: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='\r')

        try:
            # Run Inference
            results = model.detect(frame, conf=0.10)
            
            # Convert to Detections
            detections = sv.Detections.from_inference(results)

            # Logic
            class_names = [model.names[id].lower() for id in detections.class_id]
            if len(class_names) > 0:
                print(f"\n[DEBUG] Frame {frame_count}: Detected {len(class_names)} objects: {set(class_names)}")
            
            person_indices = [i for i, name in enumerate(class_names) if name == 'person']
            equip_indices = [i for i, name in enumerate(class_names) if name in HEIGHT_EQUIPMENT]

            person_detections = detections[person_indices]
            equipment_detections = detections[equip_indices]

            final_labels = []
            for p_box in person_detections.xyxy:
                at_height = False
                px1, py1, px2, py2 = p_box
                p_area = (px2 - px1) * (py2 - py1)
                
                for e_box in equipment_detections.xyxy:
                    ex1, ey1, ex2, ey2 = e_box
                    ix1, iy1 = max(px1, ex1), max(py1, ey1)
                    ix2, iy2 = min(px2, ex2), min(py2, ey2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if p_area > 0 and (inter_area / p_area) > 0.15:
                            at_height = True
                            break
                final_labels.append("Person at Height" if at_height else "Person on Ground")

            # Annotate
            annotated_frame = box_annotator.annotate(scene=frame, detections=equipment_detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=person_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=person_detections, 
                labels=final_labels
            )
            
            # Save to RAM
            processed_frames_buffer.append(annotated_frame)

        except Exception as e:
            print(f"\nError processing frame {frame_count}: {e}")
            processed_frames_buffer.append(frame)

    cap.release()
    print("\nProcessing Complete! Launching Player...")
    print("Press 'q' to quit.")

    # --- PLAYBACK LOOP ---
    # This acts as the "popup" player
    cv2.namedWindow("Work at Height Safety (Playback)", cv2.WINDOW_NORMAL)
    
    while True:
        for frame in processed_frames_buffer:
            cv2.imshow("Work at Height Safety (Playback)", frame)
            
            # 30 FPS delay
            if cv2.waitKey(33) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()