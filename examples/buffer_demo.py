"""
Detectron2 demo with Buffer module for temporal context.

This demo shows how to use the Buffer module to maintain temporal context
of frames and results, which can be used for smoothing, interpolation, or
other temporal processing.
"""

import cv2
import time
import torch
import pixelflow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from pixelflow.buffer import Buffer
from pixelflow.results import from_detectron2

print("Setting up Detectron2 with PixelFlow Buffer...")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.DEVICE = device

print("Creating predictor...")
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.get("thing_classes", None)

# Initialize Buffer with 5 frames
# This will give us 2 past frames, 1 current, 2 future frames
buffer = Buffer(frames=5)

cap = cv2.VideoCapture("data/crowd.mp4")

if not cap.isOpened():
    print("Error: Cannot open crowd.mp4")
    exit(1)

prev_time = 0
frame_count = 0
buffer_filled_at = 0

print("🚀 Detectron2 + PixelFlow Buffer Demo")
print(f"Buffer size: {buffer.buffer_size} frames")
print(f"Buffer delay: {buffer.delay} frames")
print("Press 'q' to quit")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        buffer_filled_at = 0
        buffer.reset()
        continue

    frame_count += 1
    
    # Run inference on current frame
    outputs = predictor(frame)
    results = from_detectron2(outputs)
    
    if class_names:
        for pred in results.detections:
            if pred.class_id is not None and pred.class_id < len(class_names):
                pred.class_name = class_names[pred.class_id]
    
    # Buffer the frame and results
    # Returns middle frame/results once buffer is full
    results, frame = buffer.update(results, frame)
    
    # Track when buffer becomes full
    if buffer.is_full and buffer_filled_at == 0:
        buffer_filled_at = frame_count
        print(f"Buffer filled at frame {frame_count}, now showing frame {frame_count - buffer.delay}")
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    # Annotate the frame with detections
    frame = pixelflow.annotate.box(frame, results, thickness=2)
    frame = pixelflow.annotate.label(frame, results)
    
    # Create info text with buffer status
    info_text = [
        f"FPS: {fps:.1f} | Frame: {frame_count}",
        f"Buffer: {'FILLING' if not buffer.is_full else 'ACTIVE'} ({buffer.current_size}/{buffer.buffer_size})",
    ]
    
    if buffer.is_full:
        info_text.append(f"Showing frame: {frame_count - buffer.delay} (delay: {buffer.delay})")
    else:
        info_text.append(f"Buffering... {buffer.current_size}/{buffer.buffer_size} frames")
    
    info_text.extend([
        "",
        f"Detections: {len(results.detections) if results else 0} objects"
    ])
    
    # Draw info panel
    overlay = frame.copy()
    panel_height = len(info_text) * 25 + 20
    cv2.rectangle(overlay, (5, 5), (350, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 25
    for i, text in enumerate(info_text):
        # Highlight buffer status
        if i == 1:
            color = (0, 255, 255) if buffer.is_full else (0, 165, 255)
        elif i == 2:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        cv2.putText(frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    
    cv2.imshow('Detectron2 + Buffer Demo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print buffer statistics
print(f"\nBuffer Statistics:")
print(f"Total frames processed: {buffer.frames_processed}")
print(f"Buffer became full at frame: {buffer_filled_at}")
print(f"Effective delay: {buffer.delay} frames")

cap.release()
cv2.destroyAllWindows()