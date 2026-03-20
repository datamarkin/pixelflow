"""
Detectron2 demo with PixelFlow features - tracking, zones, lines, and enhanced annotations.
"""

import cv2
import time
import torch
import pixelflow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from pixelflow.tracker import ByteTracker
from pixelflow.zones import Zones
from pixelflow.crossings import Crossings
from pixelflow.detections import from_detectron2

print("Setting up Detectron2 with PixelFlow features...")

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

tracker = ByteTracker(
    track_activation_threshold=0.25,
    lost_track_buffer=60,
    minimum_matching_threshold=0.8,
    minimum_consecutive_frames=2,
    second_match_threshold=0.2,
    assignment_threshold=0.5
)

zones = Zones()
crossings = Crossings()

zones.add_zone(
    polygon=[(50, 200), (350, 200), (350, 480), (50, 480)],
    zone_id="entrance_zone",
    name="Entrance",
    color=(0, 255, 0),
    trigger_strategy="center"
)

crossings.add_line(
    start=(375, 150),
    end=(375, 480),
    line_id="divider_line",
    name="Field Divider",
    color=(255, 255, 0),
    triggering_anchor="center",
    minimum_crossing_threshold=1
)

cap = cv2.VideoCapture("data/people.mp4")

if not cap.isOpened():
    print("Error: Cannot open crowd.mp4")
    exit(1)

prev_time = 0
frame_count = 0

print("🚀 Detectron2 + PixelFlow Demo")
print("Press 'q' to quit")
print()

process_every_n_frames = 1

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue


    # Convert to RGB for pixelflow; keep BGR copy for Detectron2
    bgr_frame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_count += 1
    
    if process_every_n_frames > 1 and frame_count % process_every_n_frames != 0:
        cv2.imshow('Detectron2 + PixelFlow Demo', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    outputs = predictor(bgr_frame)
    results = from_detectron2(outputs)
    
    if class_names:
        for pred in results.detections:
            if pred.class_id is not None and pred.class_id < len(class_names):
                pred.class_name = class_names[pred.class_id]
    
    results = tracker.update(results)
    results = zones.update(results)
    results = crossings.update(results)
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    frame = pixelflow.annotate.zones(
        frame,
        zones,
        opacity=0.2,
        border_thickness=2,
        show_counts=True,
        show_names=True
    )
    
    # frame = pixelflow.annotate.line_zones(frame, lines)
    
    frame = pixelflow.annotate.box(frame, results, thickness=2)
    
    # frame = pixelflow.annotate.footprint(frame, results)
    
    for pred in results.detections:
        if pred.zones:
            pred.zone_info = f"[{', '.join(pred.zone_names)}]"
        else:
            pred.zone_info = ""
    
    # frame = pixelflow.annotate.label(frame, results)
    
    metrics = tracker.get_metrics()
    zone_counts = zones.get_zone_counts()
    line_counts = lines.get_line_counts()
    
    info_text = [
        f"FPS: {fps:.1f} | Frame: {frame_count}",
        f"Active Tracks: {metrics['active_tracks']} | Total: {metrics['total_tracks']}",
        f"Detection: {len(results.detections)} objects",
        "",
        "Zone Counts:"
    ]
    
    for zone_id, count in zone_counts.items():
        zone = zones.get_zone(zone_id)
        info_text.append(f"  {zone.name}: {count}")
    
    if line_counts:
        info_text.append("")
        info_text.append("Line Crossings:")
        for line_id, counts in line_counts.items():
            line_name = counts['name']
            in_count = counts['in_count']
            out_count = counts['out_count']
            info_text.append(f"  {line_name}: In={in_count}, Out={out_count}")
    
    
    overlay = frame.copy()
    panel_height = len(info_text) * 25 + 20
    cv2.rectangle(overlay, (5, 5), (400, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    y_offset = 25
    for i, text in enumerate(info_text):
        color = (0, 255, 0) if i < 3 else (255, 255, 255)
        cv2.putText(frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    
    cv2.imshow('Detectron2 + PixelFlow Demo', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
