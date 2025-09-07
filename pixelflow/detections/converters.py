"""
Detection Converters for Machine Learning Framework Integration.

Provides standardized conversion utilities to transform detection outputs from 
various machine learning frameworks (Detectron2, Ultralytics YOLO, Datamarkin API) 
into PixelFlow's unified Detections format for consistent processing and visualization.
"""

import ast
import cv2
import numpy as np
from typing import (List, Dict, Any, Union)

__all__ = ["from_datamarkin_api", "from_detectron2", "from_ultralytics", "from_transformers", "from_sam", "from_datamarkin_csv"]


def from_datamarkin_api(api_response: Dict[str, Any]):
    """
    Convert Datamarkin API response to a unified Detections object.
    
    Args:
        api_response (Dict[str, Any]): The API response containing predictions
                                      with objects containing bbox, mask, keypoints,
                                      class, and bbox_score fields.
        
    Returns:
        Detections: Unified Detections object containing all detected objects
                   with standardized bbox, mask, keypoints, and confidence data.
    
    Example:
        >>> import pixelflow as pf
        >>> # Datamarkin API response from object detection service
        >>> api_response = {
        ...     "predictions": {
        ...         "objects": [
        ...             {
        ...                 "bbox": [100, 50, 200, 150],
        ...                 "mask": [[110, 60], [190, 140]],
        ...                 "class": "person",
        ...                 "bbox_score": 0.85
        ...             }
        ...         ]
        ...     }
        ... }
        >>> detections = pf.detections.from_datamarkin_api(api_response)
        >>> print(f"Found {len(detections.detections)} objects")
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Create the Detection object
        detection = Detection(
            bbox=bbox,
            masks=mask,
            keypoints=keypoints,
            class_id=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        detections_obj.add_detection(detection)

    return detections_obj


def from_detectron2(detectron2_results: Dict[str, Any]):
    """
    Convert Detectron2 inference results to a unified Detections object.
    
    Extracts bounding boxes, confidence scores, class IDs, segmentation masks, 
    and keypoints from Detectron2's instances format and standardizes them 
    into PixelFlow's Detection objects.
    
    Args:
        detectron2_results (Dict[str, Any]): Detectron2 inference results containing
                                           'instances' with prediction data including
                                           pred_boxes, scores, pred_classes, pred_masks,
                                           and pred_keypoints.
        
    Returns:
        Detections: Unified Detections object with all detected instances converted
                   to standardized format with XYXY bounding boxes and binary masks.
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from detectron2 import model_zoo
        >>> from detectron2.engine import DefaultPredictor
        >>> from detectron2.config import get_cfg
        >>> 
        >>> # Setup Detectron2 model
        >>> cfg = get_cfg()
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        >>> predictor = DefaultPredictor(cfg)
        >>> 
        >>> # Run inference and convert
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = predictor(image)  # Raw Detectron2 output
        >>> detections = pf.detections.from_detectron2(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Access standardized detection data
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_id}, Confidence: {detection.confidence}")
    
    Notes:
        - Bounding boxes are converted from Detectron2's tensor format to XYXY lists
        - Segmentation masks are converted to boolean numpy arrays
        - All tensor data is moved to CPU for processing
        - Keypoints are extracted but conversion to PixelFlow format needs implementation
    """
    from .detections import Detections, Detection
    
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


# TODO check/verify & improve the mask part
def from_ultralytics(ultralytics_results: Union[Any, List[Any]]):
    """
    Convert Ultralytics YOLO results to a unified Detections object.
    
    Supports both detection and segmentation models, handling bounding boxes,
    confidence scores, class IDs, segmentation masks, and tracker IDs.
    Automatically processes letterbox padding removal and mask resizing to
    original image dimensions.
    
    Args:
        ultralytics_results (Union[Any, List[Any]]): YOLO results from Ultralytics
                                                    library, either single result
                                                    object or list of results.
        
    Returns:
        Detections: Unified Detections object containing all detected objects with
                   standardized XYXY bounding boxes, binary masks, polygon segments,
                   and tracker IDs if available.
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load YOLO model and run inference
        >>> model = YOLO("yolov8n.pt")
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw YOLO output
        >>> detections = pf.detections.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Access detection data
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_name}, Confidence: {detection.confidence:.2f}")
        >>> 
        >>> # With segmentation model
        >>> seg_model = YOLO("yolov8n-seg.pt")
        >>> outputs = seg_model.predict(image)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> 
        >>> # With tracking
        >>> outputs = model.track(image, tracker="bytetrack.yaml")
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for detection in detections.detections:
        ...     if detection.tracker_id is not None:
        ...         print(f"Object {detection.tracker_id}: {detection.class_name}")
    
    Notes:
        - Handles letterbox padding removal automatically for accurate mask sizing
        - Binary masks are resized to original image dimensions using nearest interpolation
        - Supports both polygon segments and binary mask formats
        - Tracker IDs are extracted when available from model.track() calls
        - Original YOLO mask data is preserved in _ultralytics_masks for reference
        
    Performance Notes:
        - Uses efficient tensor operations for batch processing
        - Minimizes CPU/GPU transfers by processing all boxes at once
        - Mask processing is optimized with OpenCV resize operations
    """
    from .detections import Detections, Detection
    
    detections_obj = Detections()
    
    # Handle empty results
    if not ultralytics_results:
        return detections_obj
    
    # Handle both single result and list of results
    if isinstance(ultralytics_results, list):
        # Get the first result (YOLO returns a list with one result per image)
        result = ultralytics_results[0]
    else:
        # Already a single result object
        result = ultralytics_results
    
    # Handle case where there are no detections
    if result.boxes is None or len(result.boxes) == 0:
        return detections_obj
    
    # Get all box data in one tensor transfer (more efficient)
    boxes_data = result.boxes.data.cpu().numpy()
    
    # Extract components from the tensor
    # Format: [x1, y1, x2, y2, conf, class_id, ...] or [x1, y1, x2, y2, conf, class_id, track_id]
    xyxy = boxes_data[:, :4]  # Bounding boxes
    confidences = boxes_data[:, 4]  # Confidence scores  
    class_ids = boxes_data[:, 5].astype(int)  # Class IDs
    
    # Check if tracker IDs are available (when using model.track())
    tracker_ids = None
    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
        tracker_ids = result.boxes.id.cpu().numpy().astype(int)
    
    # Check if we have segmentation masks
    has_masks = hasattr(result, 'masks') and result.masks is not None
    
    # Process each detection
    num_detections = len(xyxy)
    
    # Get binary masks if available (shape: [num_masks, height, width])
    binary_masks = None
    orig_shape = None
    if has_masks and hasattr(result.masks, 'data'):
        # Convert masks to numpy arrays
        binary_masks = result.masks.data.cpu().numpy()
        # Get original image shape from masks or result
        if hasattr(result.masks, 'orig_shape'):
            orig_shape = result.masks.orig_shape  # (height, width)
        elif hasattr(result, 'orig_shape'):
            orig_shape = result.orig_shape  # (height, width)
    
    for i in range(num_detections):
        # Basic detection info
        bbox = xyxy[i].tolist()
        confidence = float(confidences[i])
        class_id = int(class_ids[i])
        
        # Extract class name from result if available
        class_name = None
        if hasattr(result, 'names') and result.names:
            if class_id in result.names:
                class_name = result.names[class_id]
        
        # Get tracker ID if available
        tracker_id = None
        if tracker_ids is not None:
            tracker_id = int(tracker_ids[i])
        
        # Handle masks if available
        masks = None
        segments = None
        if has_masks:
            # Store polygon format (xy) for segments
            segments = result.masks.xy[i]
            if segments is not None and len(segments) > 0:
                segments = segments.astype(int).tolist()
            
            # Store binary mask if available
            if binary_masks is not None:
                # Get the binary mask for this detection
                mask = binary_masks[i]
                
                # Handle letterbox padding and resize mask to original shape
                if orig_shape is not None and mask.shape[:2] != orig_shape:
                    # YOLO uses letterboxing: it pads to square then resizes
                    # We need to remove padding and resize to original dimensions
                    mask_h, mask_w = mask.shape[:2]  # Should be 640x640
                    orig_h, orig_w = orig_shape  # Original image dimensions
                    
                    # Calculate the scale and padding used by YOLO
                    scale = min(mask_h / orig_h, mask_w / orig_w)
                    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                    
                    # Calculate padding
                    pad_h = (mask_h - new_h) // 2
                    pad_w = (mask_w - new_w) // 2
                    
                    # Remove padding
                    mask = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
                    
                    # Resize to original dimensions
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (orig_w, orig_h),  # cv2 uses (width, height)
                                    interpolation=cv2.INTER_NEAREST)
                
                mask = mask.astype(bool)
                masks = [mask]  # Wrap in list for consistency with API
            elif segments is not None:
                # Fallback to polygon format if binary not available
                masks = [segments]
        
        # Create detection object
        detection = Detection(
            bbox=bbox,
            masks=masks,  # Can be either binary mask or polygon coordinates
            segments=segments,  # Always polygon coordinates
            keypoints=None,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            tracker_id=tracker_id
        )
        
        detections_obj.add_detection(detection)
    
    # Store the original YOLO masks data for later use if needed
    # This avoids processing masks until they're actually used
    if has_masks:
        detections_obj._ultralytics_masks = result.masks
    
    return detections_obj


def from_transformers(transformers_results):
    pass


def from_sam(sam_results):
    pass


def from_datamarkin_csv(group: Any, height: int, width: int):
    """
    Convert CSV data from Datamarkin format to a unified Detections object.
    
    Processes normalized coordinates from CSV format and converts them to pixel
    coordinates using the provided image dimensions. Handles both bounding box
    and segmentation polygon data.
    
    Args:
        group (Any): Pandas DataFrame group containing CSV rows with columns:
                    'xmin', 'ymin', 'xmax', 'ymax', 'segmentation', 'class',
                    and optional 'confidence'.
        height (int): Image height in pixels to denormalize coordinates.
        width (int): Image width in pixels to denormalize coordinates.
        
    Returns:
        Detections: Unified Detections object with pixel coordinates converted
                   from normalized values, including bounding boxes and polygon masks.
    
    Example:
        >>> import pandas as pd
        >>> import pixelflow as pf
        >>> 
        >>> # Load CSV data with normalized coordinates
        >>> df = pd.read_csv("annotations.csv")
        >>> # Group by image if processing multiple images
        >>> for image_name, group in df.groupby('image'):
        ...     detections = pf.detections.from_datamarkin_csv(group, height=480, width=640)
        ...     print(f"Image {image_name}: {len(detections.detections)} objects")
        >>> 
        >>> # Single image processing
        >>> detections = pf.detections.from_datamarkin_csv(df, height=1080, width=1920)
    
    Notes:
        - Input coordinates must be normalized (0.0-1.0 range)
        - Segmentation data is expected as string representation of coordinate lists
        - Confidence values are optional and will be None if not provided
        - Polygon coordinates are converted to tuples for consistent formatting
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    for index, row in group.iterrows():
        # Get the bounding box coordinates and denormalize them
        xmin = int(row['xmin'] * width)
        ymin = int(row['ymin'] * height)
        xmax = int(row['xmax'] * width)
        ymax = int(row['ymax'] * height)

        # Convert normalized points to pixel coordinates for the mask
        segmentation_list = ast.literal_eval(row['segmentation'])
        segmentation_points = []
        for i in range(0, len(segmentation_list), 2):
            x = int(segmentation_list[i] * width)
            y = int(segmentation_list[i + 1] * height)
            segmentation_points.append((x, y))  # Convert to tuple for polygon points

        # Create the Detection object
        detection = Detection(
            bbox=[xmin, ymin, xmax, ymax],
            masks=[segmentation_points],  # Add mask as list of lists of tuples
            keypoints=None,  # TODO
            class_id=row['class'],
            confidence=row.get('confidence', None)  # Add confidence if available
        )

        # Add the prediction to the predictions list
        detections_obj.add_detection(detection)

    return detections_obj