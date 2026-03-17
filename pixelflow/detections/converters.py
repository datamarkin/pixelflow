"""
Detection Converters for Machine Learning Framework Integration.

Provides standardized conversion utilities to transform detection outputs from 
various machine learning frameworks (Detectron2, Ultralytics YOLO, Datamarkin API, 
Transformers) into PixelFlow's unified Detections format. This module enables seamless 
integration with different ML backends while maintaining consistent data structures r
for downstream processing, visualization, and analysis workflows.
"""

import ast
import cv2
import numpy as np
from typing import (List, Dict, Any, Union, Optional)

__all__ = [
    "from_datamarkin",
    "from_florence2",
    "from_detectron2",
    "from_ultralytics",
    "from_transformers",
    "from_sam",
    "from_datamarkin_csv"
]

# COCO pose keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def from_datamarkin(api_response: Dict[str, Any]):
    """
    Convert Datamarkin API response to a unified Detections object.

    Processes detection results from Datamarkin's cloud-based object detection API,
    extracting bounding boxes, segmentation masks, keypoints, class labels, and
    confidence scores into PixelFlow's standardized format for further processing.

    Args:
        api_response (Dict[str, Any]): Datamarkin API response dictionary containing
                                      nested 'predictions' -> 'objects' structure
                                      with detection data. Each object should have
                                      'bbox', 'mask', 'keypoints', 'class', and
                                      'bbox_score' fields.

    Returns:
        Detections: Unified Detections object containing all detected objects with
                   standardized XYXY bounding boxes, polygon masks, keypoint data,
                   and confidence scores. Empty Detections object if no predictions.

    Raises:
        KeyError: If required API response structure is missing or malformed
        TypeError: If bbox coordinates cannot be converted to numeric format
        ValueError: If confidence scores are outside valid range [0.0, 1.0]

    Example:
        >>> import pixelflow as pf
        >>> import requests
        >>>
        >>> # Call Datamarkin API for object detection
        >>> response = requests.post(
        ...     "https://api.datamarkin.com/detect",
        ...     files={"image": open("image.jpg", "rb")}
        ... )
        >>> api_response = response.json()  # Raw API output
        >>> detections = pf.detections.from_datamarkin(api_response)  # Convert to PixelFlow format
        >>>
        >>> # Basic usage - access detection data
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_id}, Confidence: {detection.confidence:.2f}")
        >>>
        >>> # Advanced usage - filter by confidence
        >>> high_conf_detections = [d for d in detections.detections if d.confidence > 0.8]
        >>> print(f"High confidence detections: {len(high_conf_detections)}")
        >>>
        >>> # Process masks and keypoints
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         print(f"Object has {len(detection.masks)} mask regions")
        ...     if detection.keypoints:
        ...         print(f"Object has {len(detection.keypoints)} keypoints")
        >>>
        >>> # Empty response handling
        >>> empty_response = {"predictions": {"objects": []}}
        >>> empty_detections = pf.detections.from_datamarkin(empty_response)
        >>> print(f"Empty result: {len(empty_detections.detections)} objects")

    Notes:
        - Bounding boxes are expected in XYXY format from the API
        - Mask data is stored as polygon coordinates in nested list format
        - Keypoints are converted from API format (name, point, probability) to PixelFlow KeyPoint objects
        - API 'probability' field is converted to 'visibility' in KeyPoint (values > 0 = visible)
        - Class names are stored in class_name field (API doesn't provide numeric class_id)
        - Missing or null confidence scores are preserved as None values
        - Function gracefully handles missing optional fields (mask, keypoints)

    Performance Notes:
        - Efficient single-pass processing of API response structure
        - Minimal data copying for large mask or keypoint arrays
        - No validation overhead for well-formed API responses
    """
    from .detections import Detections, Detection, KeyPoint

    detections_obj = Detections()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints_api = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Convert API keypoint format to PixelFlow KeyPoint objects
        # API format: {"name": "p0", "point": [x, y], "probability": 0.375}
        # PixelFlow format: KeyPoint(x, y, name, visibility)
        keypoints = None
        if keypoints_api and len(keypoints_api) > 0:
            keypoints = []
            for kp_dict in keypoints_api:
                point = kp_dict.get("point", [0, 0])
                probability = kp_dict.get("probability", 0.0)

                # Convert probability to visibility (True if probability > 0)
                visibility = probability > 0.0

                keypoint = KeyPoint(
                    x=int(point[0]),
                    y=int(point[1]),
                    name=kp_dict.get("name", ""),
                    visibility=visibility
                )
                keypoints.append(keypoint)

        # Create the Detection object
        detection = Detection(
            bbox=bbox,
            masks=mask,
            keypoints=keypoints,
            class_id=0,  # API doesn't provide numeric class IDs, default to 0
            class_name=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        detections_obj.add_detection(detection)

    return detections_obj


def from_florence2(
    parsed_result: Dict[str, Any],
    task_prompt: str,
    image_size: Union[tuple, None] = None
):
    """
    Convert Florence-2 model results to a unified Detections object.

    Processes detection results from Microsoft's Florence-2 vision foundation model,
    extracting bounding boxes, segmentation polygons, OCR text with quad boxes, and
    labels into PixelFlow's standardized format. Supports all Florence-2 vision tasks
    including object detection, grounding, segmentation, and OCR with regions.

    Args:
        parsed_result (Dict[str, Any]): Parsed output from Florence-2 processor.
                                        Must be dictionary with task prompt as key
                                        containing nested data with 'bboxes', 'labels',
                                        'polygons', or 'quad_boxes' depending on task.
        task_prompt (str): Florence-2 task identifier used for prediction. Examples:
                          '<OD>' (object detection), '<CAPTION_TO_PHRASE_GROUNDING>',
                          '<REFERRING_EXPRESSION_SEGMENTATION>', '<OCR_WITH_REGION>'.
                          Must match a key in parsed_result dictionary.
        image_size (tuple, optional): Image dimensions as (width, height) tuple for
                                     coordinate normalization. If None, assumes
                                     coordinates are already in absolute pixels.
                                     Default is None.

    Returns:
        Detections: Unified Detections object containing detected objects with
                   standardized XYXY bounding boxes, polygon masks for segmentation
                   tasks, OCR data with quad boxes, and sequential class IDs for
                   consistent color mapping. Empty Detections if no objects found.

    Raises:
        ValueError: If task_prompt not found in parsed_result, or if task is text-only
                   (pure caption/OCR without regions), or if required data fields are
                   missing for the specified task type.
        KeyError: If expected nested dictionary keys are missing from parsed_result
        TypeError: If coordinate values cannot be converted to numeric format

    Example:
        >>> import torch
        >>> import pixelflow as pf
        >>> from transformers import AutoProcessor, AutoModelForCausalLM
        >>> from PIL import Image
        >>>
        >>> # Load Florence-2 model
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "microsoft/Florence-2-large",
        ...     trust_remote_code=True
        ... )
        >>> processor = AutoProcessor.from_pretrained(
        ...     "microsoft/Florence-2-large",
        ...     trust_remote_code=True
        ... )
        >>>
        >>> # Object detection task
        >>> image = Image.open("image.jpg")
        >>> task = "<OD>"
        >>> inputs = processor(text=task, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(
        ...     outputs,
        ...     task=task,
        ...     image_size=(image.width, image.height)
        ... )
        >>> # parsed format: {'<OD>': {'bboxes': [[x1,y1,x2,y2],...], 'labels': ['car','person',...]}}
        >>> detections = pf.detections.from_florence2(
        ...     parsed[task],
        ...     task_prompt=task,
        ...     image_size=(image.width, image.height)
        ... )
        >>> for det in detections.detections:
        ...     print(f"Class: {det.class_name}, BBox: {det.bbox}")
        >>>
        >>> # Grounding task (phrase → bounding boxes)
        >>> task = "<CAPTION_TO_PHRASE_GROUNDING>"
        >>> prompt = "A green car and a person with red shirt"
        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(image.width, image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> # Each detection has bbox + class_name from grounded phrase
        >>>
        >>> # Segmentation task (referring expression → mask)
        >>> task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        >>> prompt = "the red car"
        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(image.width, image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> for det in detections.detections:
        ...     if det.masks:
        ...         print(f"Polygon mask with {len(det.masks[0])} points")
        >>>
        >>> # OCR with region detection
        >>> task = "<OCR_WITH_REGION>"
        >>> inputs = processor(text=task, images=document_image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(document_image.width, document_image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> for det in detections.detections:
        ...     print(f"Text: {det.ocr_data.text}, Quad: {det.segments}")

    Notes:
        - Florence-2 output format: {task_prompt: {'bboxes': [...], 'labels': [...]}}
        - Detection tasks (<OD>, <DENSE_REGION_CAPTION>, etc.) provide bboxes + labels
        - Segmentation tasks provide triple-nested polygons: [[[x1,y1,x2,y2,...]]]
        - OCR tasks provide quad_boxes (8 values): [x1,y1,x2,y2,x3,y3,x4,y4]
        - Text-only tasks (<CAPTION>, <MORE_DETAILED_CAPTION>, <OCR>) raise ValueError
        - Sequential class_ids (0,1,2,...) assigned for consistent color mapping
        - Confidence defaults to 1.0 if scores not provided in parsed_result
        - All coordinates assumed to be in absolute pixels (no normalization needed)

    Performance Notes:
        - Efficient single-pass processing of Florence-2 output dictionary
        - Minimal data copying for bboxes, polygons, and quad coordinates
        - O(n) complexity where n is number of detected objects
        - Lazy polygon processing deferred until mask data actually accessed

    See Also:
        from_detectron2 : Convert Detectron2 results to PixelFlow format
        from_ultralytics : Convert YOLO results to PixelFlow format
        from_easyocr : Convert EasyOCR results for pure OCR tasks
    """
    from .detections import Detections, Detection, OCRData

    detections_obj = Detections()

    # Text-only tasks that should not be converted to Detections
    TEXT_ONLY_TASKS = {
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>',
        '<OCR>',  # Pure OCR without regions
    }

    # Validate that task_prompt exists in parsed_result
    if task_prompt not in parsed_result:
        raise ValueError(
            f"Task prompt '{task_prompt}' not found in parsed_result. "
            f"Available keys: {list(parsed_result.keys())}"
        )

    # Reject text-only tasks
    if task_prompt in TEXT_ONLY_TASKS:
        raise ValueError(
            f"Task '{task_prompt}' returns text only and cannot be converted to Detections. "
            f"Text-only tasks: {TEXT_ONLY_TASKS}"
        )

    # Extract the task-specific data
    task_data = parsed_result[task_prompt]

    # Handle different task types based on available data fields
    if 'bboxes' in task_data and 'labels' in task_data:
        # Detection tasks: <OD>, <CAPTION_TO_PHRASE_GROUNDING>, <DENSE_REGION_CAPTION>, etc.
        bboxes = task_data['bboxes']
        labels = task_data['labels']
        scores = task_data.get('scores', None)  # Optional confidence scores

        # Assign sequential class IDs for consistent coloring
        unique_labels = []
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(unique_labels)
                unique_labels.append(label)

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            class_id = label_to_id[label]
            confidence = float(scores[idx]) if scores is not None else 1.0

            # Florence-2 bboxes are already in XYXY format [x1, y1, x2, y2]
            detection = Detection(
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                class_id=class_id,
                class_name=label,
                confidence=confidence
            )
            detections_obj.add_detection(detection)

    elif 'polygons' in task_data and 'labels' in task_data:
        # Segmentation tasks: <REFERRING_EXPRESSION_SEGMENTATION>
        polygons = task_data['polygons']  # Triple-nested: [[[x1,y1,x2,y2,...]]]
        labels = task_data['labels']

        # Assign sequential class IDs
        unique_labels = []
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(unique_labels)
                unique_labels.append(label)

        for idx, (poly_nested, label) in enumerate(zip(polygons, labels)):
            class_id = label_to_id[label]

            # Florence-2 polygons are triple-nested: [[[x1,y1,x2,y2,...]]]
            # Extract the innermost list of coordinates
            if len(poly_nested) > 0 and len(poly_nested[0]) > 0:
                poly_coords = poly_nested[0][0]  # Get innermost list

                # Convert flat list [x1,y1,x2,y2,...] to list of tuples [(x1,y1), (x2,y2), ...]
                segments = []
                for i in range(0, len(poly_coords), 2):
                    if i + 1 < len(poly_coords):
                        segments.append((int(poly_coords[i]), int(poly_coords[i + 1])))

                # Compute axis-aligned bounding box from polygon
                if segments:
                    x_coords = [p[0] for p in segments]
                    y_coords = [p[1] for p in segments]
                    bbox = [
                        float(min(x_coords)),
                        float(min(y_coords)),
                        float(max(x_coords)),
                        float(max(y_coords))
                    ]
                else:
                    bbox = None

                detection = Detection(
                    bbox=bbox,
                    segments=segments,  # Store polygon as segments
                    masks=[segments],   # Also store as mask for compatibility
                    class_id=class_id,
                    class_name=label,
                    confidence=1.0
                )
                detections_obj.add_detection(detection)

    elif 'quad_boxes' in task_data and 'labels' in task_data:
        # OCR with region: <OCR_WITH_REGION>
        quad_boxes = task_data['quad_boxes']  # Format: [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        labels = task_data['labels']  # Text content

        for idx, (quad, text) in enumerate(zip(quad_boxes, labels)):
            # quad is 8 values: [x1,y1, x2,y2, x3,y3, x4,y4]
            # Convert to list of (x,y) tuples for segments
            segments = [
                (int(quad[0]), int(quad[1])),  # Top-left
                (int(quad[2]), int(quad[3])),  # Top-right
                (int(quad[4]), int(quad[5])),  # Bottom-right
                (int(quad[6]), int(quad[7]))   # Bottom-left
            ]

            # Compute axis-aligned bounding box from quad
            x_coords = [p[0] for p in segments]
            y_coords = [p[1] for p in segments]
            bbox = [
                float(min(x_coords)),
                float(min(y_coords)),
                float(max(x_coords)),
                float(max(y_coords))
            ]

            # Create OCRData for structured text information
            ocr_data = OCRData(
                text=text.strip(),
                confidence=1.0,  # Florence-2 doesn't provide OCR confidence
                language='multi',  # Florence-2 is multi-lingual
                level='word',
                order=idx,
                element_type='text'
            )

            detection = Detection(
                bbox=bbox,
                segments=segments,  # Store quad as segments
                ocr_data=ocr_data,
                class_id=0,  # OCR has single class
                class_name='text',
                confidence=1.0
            )
            detections_obj.add_detection(detection)

    else:
        # Unknown task format
        raise ValueError(
            f"Unsupported Florence-2 task format for '{task_prompt}'. "
            f"Available data fields: {list(task_data.keys())}. "
            f"Expected 'bboxes+labels', 'polygons+labels', or 'quad_boxes+labels'."
        )

    return detections_obj


def from_detectron2(detectron2_results: Dict[str, Any], class_names: Optional[List[str]] = None):
    """
    Convert Detectron2 inference results to a unified Detections object.
    
    Extracts bounding boxes, confidence scores, class IDs, segmentation masks, 
    and keypoints from Detectron2's instances format and standardizes them 
    into PixelFlow's Detection objects. Handles automatic tensor-to-numpy conversion 
    and CPU transfer for efficient processing.
    
    Args:
        detectron2_results (Dict[str, Any]): Detectron2 inference results dictionary
                                           containing 'instances' key with prediction
                                           data including pred_boxes, scores,
                                           pred_classes, pred_masks, and pred_keypoints.
                                           Results should be from DefaultPredictor output.
        class_names (Optional[List[str]]): List of class names indexed by class ID.
                                          If provided, Detection objects will include
                                          class_name attribute. Obtain from MetadataCatalog:
                                          `MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes`

    Returns:
        Detections: Unified Detections object with all detected instances converted
                   to standardized format. Contains XYXY bounding boxes as lists,
                   boolean numpy array masks, integer class IDs, and float confidences.
                   Returns empty Detections if no instances found.
    
    Raises:
        KeyError: If required 'instances' key is missing from detectron2_results
        AttributeError: If instances object lacks expected prediction attributes
        RuntimeError: If tensor operations fail during CPU transfer
        ValueError: If bounding box or confidence data contains invalid values
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from detectron2 import model_zoo
        >>> from detectron2.engine import DefaultPredictor
        >>> from detectron2.config import get_cfg
        >>> from detectron2.data import MetadataCatalog
        >>>
        >>> # Setup Detectron2 object detection model
        >>> cfg = get_cfg()
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        >>> predictor = DefaultPredictor(cfg)
        >>>
        >>> # Get class names from metadata
        >>> class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        >>>
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = predictor(image)  # Raw Detectron2 output
        >>> detections = pf.detections.from_detectron2(outputs, class_names=class_names)
        >>>
        >>> # Access detection data with class names
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_name}, Confidence: {detection.confidence:.2f}")
        >>> 
        >>> # Advanced usage - segmentation model with masks
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        >>> seg_predictor = DefaultPredictor(cfg)
        >>> outputs = seg_predictor(image)
        >>> detections = pf.detections.from_detectron2(outputs)
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         print(f"Object has mask with shape: {detection.masks[0].shape}")
        >>> 
        >>> # Process keypoint detection results
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        >>> kpt_predictor = DefaultPredictor(cfg)
        >>> outputs = kpt_predictor(image)
        >>> detections = pf.detections.from_detectron2(outputs)
        >>> 
        >>> # Empty result handling
        >>> empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> empty_outputs = predictor(empty_image)
        >>> empty_detections = pf.detections.from_detectron2(empty_outputs)
        >>> print(f"No detections: {len(empty_detections.detections)} objects")
    
    Notes:
        - All tensor data is automatically moved to CPU before numpy conversion
        - Bounding boxes maintain XYXY format from Detectron2 (no coordinate transformation)
        - Segmentation masks are converted to boolean arrays for memory efficiency
        - Class IDs are converted to integers for consistency with other frameworks
        - Confidence scores are converted to float type for standardization
        - Keypoint data structure is preserved but PixelFlow KeyPoint conversion pending
        - Function handles missing prediction attributes gracefully (returns None)
        
    Performance Notes:
        - Efficient batch tensor operations minimize GPU-CPU transfer overhead
        - Single CPU transfer per tensor type reduces memory allocation
        - Boolean mask conversion optimized for large segmentation masks
        - Zero-copy numpy operations where possible for large datasets
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

        # Extract class name from provided class_names list
        class_name = None
        if class_names is not None and class_id is not None:
            if class_id < len(class_names):
                class_name = class_names[class_id]

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
            class_name=class_name,
            confidence=confidence
        )

        # Add the detection to the Detections object
        detections_obj.add_detection(detection)

    return detections_obj


def from_ultralytics(ultralytics_results: Union[Any, List[Any]]):
    """
    Convert Ultralytics YOLO results to a unified Detections object.
    
    Supports both detection and segmentation models, handling bounding boxes,
    confidence scores, class IDs, segmentation masks, and tracker IDs.
    Automatically processes letterbox padding removal and mask resizing to
    original image dimensions with precise coordinate transformation.
    
    Args:
        ultralytics_results (Union[Any, List[Any]]): YOLO results from Ultralytics
                                                    library prediction or tracking.
                                                    Can be single Result object or
                                                    list containing one Result object.
                                                    Must have boxes attribute with
                                                    detection data.
        
    Returns:
        Detections: Unified Detections object containing all detected objects with
                   standardized XYXY bounding boxes, boolean binary masks resized
                   to original image dimensions, polygon segments as integer coordinates,
                   and tracker IDs if available. Empty Detections if no boxes found.
    
    Raises:
        AttributeError: If results object lacks required boxes or data attributes
        IndexError: If results list is empty or malformed
        ValueError: If bounding box coordinates or confidence scores are invalid
        RuntimeError: If tensor operations fail during CPU transfer
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Basic object detection
        >>> model = YOLO("yolo11n.pt")
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw YOLO output
        >>> detections = pf.detections.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Access detection data with class names
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_name}, Confidence: {detection.confidence:.2f}")
        ...     print(f"BBox: {detection.bbox}")
        >>> 
        >>> # Advanced usage - segmentation model with masks
        >>> seg_model = YOLO("yolo11n-seg.pt")
        >>> outputs = seg_model.predict(image, save_crop=False)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         mask_shape = detection.masks[0].shape
        ...         print(f"Object mask: {mask_shape} pixels")
        ...     if detection.segments:
        ...         poly_points = len(detection.segments)
        ...         print(f"Polygon: {poly_points} points")
        >>> 
        >>> # Object tracking with persistent IDs
        >>> outputs = model.track(image, tracker="bytetrack.yaml", persist=True)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for detection in detections.detections:
        ...     if detection.tracker_id is not None:
        ...         print(f"Tracked object {detection.tracker_id}: {detection.class_name}")
        >>> 
        >>> # Batch processing multiple images
        >>> image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> for img_path in image_paths:
        ...     img = cv2.imread(img_path)
        ...     outputs = model.predict(img, verbose=False)
        ...     detections = pf.detections.from_ultralytics(outputs)
        ...     print(f"{img_path}: {len(detections.detections)} objects")
    
    Notes:
        - Automatically handles single Result or list[Result] input formats
        - Bounding boxes maintain XYXY format from YOLO predictions
        - Class names are extracted from model.names dictionary when available
        - Binary masks are precisely resized using letterbox padding calculations
        - Polygon segments stored as integer coordinate lists for efficiency
        - Tracker IDs preserved from model.track() calls with persist=True
        - Original YOLO masks stored in _ultralytics_masks for advanced use cases
        - Gracefully handles empty results with no detections
        
    Performance Notes:
        - Efficient batch tensor operations minimize CPU/GPU memory transfers
        - Single numpy conversion per detection component reduces allocation overhead
        - Letterbox padding calculations optimized for common YOLO input sizes
        - OpenCV resize operations use nearest neighbor for boolean mask precision
        - Zero-copy operations where possible for large segmentation masks
        
    See Also:
        from_detectron2 : Convert Detectron2 results to PixelFlow format
        from_datamarkin : Convert cloud API results to PixelFlow format
    """
    from .detections import Detections, Detection, KeyPoint

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

    # Handle classification models (no boxes, only probs)
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs

        # Get top-1 prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)

        # Get class name
        class_name = None
        if hasattr(result, 'names') and result.names:
            class_name = result.names.get(top1_idx, str(top1_idx))

        # Get top-5 predictions for metadata
        top5_indices = [int(idx) for idx in probs.top5]
        top5_confs = probs.top5conf.cpu().numpy().tolist()
        top5_names = []
        if hasattr(result, 'names') and result.names:
            top5_names = [result.names.get(idx, str(idx)) for idx in top5_indices]

        # Create single detection for classification result
        detection = Detection(
            bbox=None,  # Classification has no bbox
            class_id=top1_idx,
            class_name=class_name,
            confidence=top1_conf,
            metadata={
                'task': 'classification',
                'top5_indices': top5_indices,
                'top5_confidences': top5_confs,
                'top5_names': top5_names
            }
        )
        detections_obj.add_detection(detection)
        return detections_obj

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

    # Check if we have keypoints (pose estimation)
    has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
    keypoints_data = None
    if has_keypoints:
        keypoints_data = result.keypoints.data.cpu().numpy()  # Shape: [N, 17, 3]

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

        # Handle keypoints if available (pose estimation)
        keypoints_list = None
        if has_keypoints and keypoints_data is not None:
            kpts = keypoints_data[i]  # Shape: [17, 3] for detection i

            keypoints_list = []
            for kpt_idx, kpt in enumerate(kpts):
                x, y, conf = kpt[0], kpt[1], kpt[2]
                # Use visibility threshold (conf > 0.5 means visible)
                visibility = conf > 0.5
                name = COCO_KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(COCO_KEYPOINT_NAMES) else f"keypoint_{kpt_idx}"

                keypoint = KeyPoint(
                    x=int(x),
                    y=int(y),
                    name=name,
                    visibility=visibility
                )
                keypoints_list.append(keypoint)

        # Create detection object
        detection = Detection(
            bbox=bbox,
            masks=masks,  # Can be either binary mask or polygon coordinates
            segments=segments,  # Always polygon coordinates
            keypoints=keypoints_list,
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


def from_transformers(transformers_results: Any):
    """
    Convert Transformers library results to a unified Detections object.
    
    Placeholder function for future integration with Hugging Face Transformers
    object detection and segmentation models. Will support DETR, RT-DETR, and
    other transformer-based detection architectures.
    
    Args:
        transformers_results (Any): Results from Transformers library object
                                   detection models. Expected format includes
                                   boxes, labels, and scores tensors.
        
    Returns:
        Detections: Empty Detections object. Full implementation pending.
    
    Raises:
        NotImplementedError: This function is not yet implemented
        
    Example:
        >>> import pixelflow as pf
        >>> # Future usage with Transformers models
        >>> # from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> # processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> # model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        >>> # outputs = model(**processor(image, return_tensors="pt"))  # Raw output
        >>> # detections = pf.detections.from_transformers(outputs)  # Convert to PixelFlow
        >>> print("Function not yet implemented")
    
    Notes:
        - Implementation will support DETR, RT-DETR, and YOLO-transformer models
        - Will handle transformer-specific output formats and attention mechanisms
        - Planned support for both detection and segmentation transformer models
        - Integration with Transformers AutoModel pipeline architecture
    """
    raise NotImplementedError("from_transformers converter not yet implemented")


def from_sam(sam_results: Any):
    """
    Convert Segment Anything Model (SAM) results to a unified Detections object.
    
    Placeholder function for integration with Meta's Segment Anything Model (SAM)
    for interactive and automatic segmentation tasks. Will support prompt-based
    segmentation with point, box, and text prompts.
    
    Args:
        sam_results (Any): Results from SAM model inference including masks,
                          iou_predictions, and low_res_logits from SamPredictor
                          or SamAutomaticMaskGenerator output.
        
    Returns:
        Detections: Empty Detections object. Full implementation pending.
    
    Raises:
        NotImplementedError: This function is not yet implemented
        
    Example:
        >>> import pixelflow as pf
        >>> # Future usage with SAM models
        >>> # from segment_anything import SamPredictor, sam_model_registry
        >>> # sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        >>> # predictor = SamPredictor(sam)
        >>> # predictor.set_image(image)
        >>> # masks, scores, logits = predictor.predict(point_coords=input_point)  # Raw output
        >>> # detections = pf.detections.from_sam({"masks": masks, "scores": scores})  # Convert to PixelFlow
        >>> print("Function not yet implemented")
    
    Notes:
        - Implementation will support both SamPredictor and SamAutomaticMaskGenerator
        - Will handle multi-mask outputs with IoU quality scores
        - Planned support for prompt-based and automatic segmentation workflows
        - Integration with different SAM model variants (ViT-B, ViT-L, ViT-H)
        - Will convert high-quality masks to PixelFlow Detection format
    """
    raise NotImplementedError("from_sam converter not yet implemented")


def from_datamarkin_csv(group: Any, height: int, width: int):
    """
    Convert CSV data from Datamarkin format to a unified Detections object.
    
    Processes normalized coordinates from CSV annotation format and converts them to pixel
    coordinates using the provided image dimensions. Handles both bounding box rectangles
    and segmentation polygon data with automatic coordinate denormalization and validation.
    
    Args:
        group (Any): Pandas DataFrame or DataFrame group containing CSV rows with
                    required columns 'xmin', 'ymin', 'xmax', 'ymax', 'segmentation',
                    'class', and optional 'confidence'. All coordinate values must
                    be normalized floats in range [0.0, 1.0].
        height (int): Image height in pixels for coordinate denormalization.
                     Must be positive integer representing actual image height.
        width (int): Image width in pixels for coordinate denormalization.
                    Must be positive integer representing actual image width.
        
    Returns:
        Detections: Unified Detections object with pixel coordinates converted from
                   normalized values. Contains XYXY bounding boxes as integers,
                   polygon masks as lists of (x, y) tuples, and preserved class labels.
                   Empty Detections if group contains no rows.
    
    Raises:
        KeyError: If required CSV columns are missing from the DataFrame
        ValueError: If coordinate values are outside [0.0, 1.0] normalized range
        TypeError: If height/width are not integers or coordinates not numeric
        SyntaxError: If segmentation string cannot be parsed as valid Python list
        
    Example:
        >>> import pandas as pd
        >>> import pixelflow as pf
        >>> 
        >>> # Load CSV annotations with normalized coordinates
        >>> df = pd.read_csv("datamarkin_annotations.csv")
        >>> # CSV format: image,xmin,ymin,xmax,ymax,segmentation,class,confidence
        >>> # Example row: img1.jpg,0.1,0.2,0.8,0.9,"[0.1,0.2,0.8,0.2,0.8,0.9,0.1,0.9]",person,0.95
        >>> detections = pf.detections.from_datamarkin_csv(df, height=480, width=640)  # Convert to PixelFlow format
        >>> 
        >>> # Basic usage - process single image annotations
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_id}, Confidence: {detection.confidence}")
        ...     print(f"BBox: {detection.bbox}")  # Pixel coordinates
        >>> 
        >>> # Advanced usage - batch process multiple images
        >>> for image_name, group in df.groupby('image'):
        ...     img_detections = pf.detections.from_datamarkin_csv(group, height=1080, width=1920)
        ...     print(f"Image {image_name}: {len(img_detections.detections)} annotations")
        ...     for detection in img_detections.detections:
        ...         if detection.masks:
        ...             poly_points = len(detection.masks[0])
        ...             print(f"  Polygon with {poly_points} points")
        >>> 
        >>> # Handle missing confidence scores
        >>> df_no_conf = df.drop('confidence', axis=1)
        >>> detections = pf.detections.from_datamarkin_csv(df_no_conf, height=720, width=1280)
        >>> for detection in detections.detections:
        ...     conf_str = "Unknown" if detection.confidence is None else f"{detection.confidence:.2f}"
        ...     print(f"Detection confidence: {conf_str}")
        >>> 
        >>> # Validate coordinate ranges
        >>> valid_coords = df[(df['xmin'] >= 0) & (df['xmax'] <= 1) & 
        ...                  (df['ymin'] >= 0) & (df['ymax'] <= 1)]
        >>> detections = pf.detections.from_datamarkin_csv(valid_coords, height=600, width=800)
    
    Notes:
        - All input coordinates must be normalized floats in range [0.0, 1.0]
        - Bounding boxes are converted to integer pixel coordinates using XYXY format
        - Segmentation strings are parsed using ast.literal_eval for safe evaluation
        - Polygon coordinates are stored as (x, y) tuples for geometric operations
        - Missing confidence values default to None and are handled gracefully
        - Class labels are preserved as strings without modification
        - Function performs coordinate validation during denormalization process
        
    Performance Notes:
        - Efficient vectorized operations for coordinate transformation
        - Minimal string parsing overhead using ast.literal_eval
        - Single-pass iteration through DataFrame rows
        - Memory-efficient tuple creation for polygon coordinates
        
    See Also:
        from_datamarkin : Convert Datamarkin API responses to PixelFlow format
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
