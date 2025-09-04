"""
Detectron2 results converter for PixelFlow

Converts Detectron2 detection results to PixelFlow's unified Detections format.
"""
from pixelflow.detections import Detections, Detection
def from_detectron2(detectron2_results):
    """
    Converts Detectron2 results to a custom Detections object.

    Args:
        detectron2_results: Detectron2 inference results containing instances with prediction data.

    Returns:
        Detections: A unified Detections object containing detections.
    """
    detections_obj = Detections()
    
    # Get instances and ensure they're on CPU for processing
    instances = detectron2_results["instances"].to("cpu")
    
    # Check if we have any instances
    if len(instances) == 0:
        return detections_obj

    # Extract prediction data
    # Bounding boxes - Detectron2 uses XYXY format
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    
    # Confidence scores
    scores = instances.scores.numpy() if instances.has("scores") else None
    
    # Class IDs  
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
    
    # Segmentation masks
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
    
    # Keypoints if available
    keypoints = None
    if instances.has("pred_keypoints"):
        keypoints = instances.pred_keypoints.numpy()

    # Iterate over each detection
    for i in range(len(instances)):
        # Extract bounding box in XYXY format
        bbox = boxes[i].tolist() if boxes is not None else None
        
        # Extract confidence score
        confidence = float(scores[i]) if scores is not None else None
        
        # Extract class ID  
        class_id = int(classes[i]) if classes is not None else None
        
        # Handle segmentation masks
        mask = None
        if masks is not None:
            mask_data = masks[i].astype(bool)
            mask = mask_data
        
        # Handle keypoints if available
        kpts = None
        if keypoints is not None:
            # Detectron2 keypoints are in format (x, y, visibility) 
            kpt_data = keypoints[i]
            # Convert to PixelFlow KeyPoint format if needed
            # This would need to be implemented based on your KeyPoint class
        
        # Create a Detection object
        detection = Detection(
            bbox=bbox,
            masks=[mask] if mask is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            confidence=confidence
        )

        # Add the detection to the Detections object
        detections_obj.add_detection(detection)

    return detections_obj