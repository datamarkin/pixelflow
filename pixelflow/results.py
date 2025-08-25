# results/__init__.py

import json
import ast
from pixelflow.validators import (validate_bbox,
                                  validate_masks,
                                  round_to_decimal,
                                  convert_datamarkin_masks,
                                  simplify_polygon)
from pixelflow.zones import Zones
from typing import (List,
                    Iterator)


# Object-oriented approach instead of a NumPy array-based approach
# Let's see how it goes


class KeyPoint:
    def __init__(self, x: int, y: int, name: str, visibility: bool):
        self.x = x
        self.y = y
        self.name = name
        self.visibility = visibility

    def to_dict(self):
        """
        Convert the KeyPoint object to a dictionary that can be easily converted to JSON.
        """
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "visibility": self.visibility
        }


class Prediction:
    def __init__(self, inference_id=None, bbox=None, masks=None, segments=None, keypoints: List[KeyPoint] = None, class_id=None,
                 class_name=None, labels=None, confidence=None, tracker_id=None, data=None, zones=None, zone_names=None,
                 line_crossings=None):
        self.inference_id = inference_id
        self.bbox = validate_bbox(bbox)
        self.masks = masks
        self.segments = segments
        self.keypoints = keypoints if keypoints is not None else None
        self.class_id = class_id
        self.class_name = class_name
        self.labels = labels
        self.confidence = round_to_decimal(confidence)
        self.tracker_id = tracker_id
        self.data = data
        self.zones = zones if zones is not None else []  # List of zone IDs
        self.zone_names = zone_names if zone_names is not None else []  # List of zone names
        self.line_crossings = line_crossings if line_crossings is not None else []  # List of line crossing events

    def to_dict(self):
        """
        Convert the Prediction object to a dictionary that can be easily converted to JSON.
        """
        return {
            "inference_id": self.inference_id,
            "bbox": self.bbox,
            "mask": self.masks,
            "segments": self.segments,
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "labels": self.labels,
            "confidence": self.confidence,
            "tracker_id": self.tracker_id,
            "data": self.data,
            "zones": self.zones,
            "zone_names": self.zone_names,
            "line_crossings": self.line_crossings
        }

    def simplify_masks(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the polygon masks using Shapely.

        Args:
            tolerance (float): The tolerance factor for simplification (higher = more simplified).
            preserve_topology (bool): If True, the function will try to preserve the polygon's topology.
        """
        if self.masks:
            # Apply the simplify function to each mask (assuming self.masks is a list of polygons)
            self.masks = [simplify_polygon(mask, tolerance, preserve_topology) for mask in self.masks]


class Results:
    def __init__(self, zones: 'Zones' = None):
        self.predictions: List[Prediction] = []
        self.zones = zones  # Store the zones object if provided

    def show(self):
        # Display the annotated image
        pass

    def add_prediction(self, prediction: Prediction):
        """
        Add a prediction to the list.
        """
        self.predictions.append(prediction)
    
    def update_zones(self, zone_manager):
        """
        Update all predictions with zone information.
        
        Args:
            zone_manager: ZoneManager instance to check zones against
            
        Returns:
            self: Returns self for method chaining
        """
        if zone_manager is not None:
            zone_manager.update(self)
        return self

    def __len__(self):
        return len(self.predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def __getitem__(self, index: int) -> Prediction:
        return self.predictions[index]

    def filter_by_confidence(self, threshold: float) -> 'Results':
        """
        Returns a new Predictions object containing only predictions
        with a confidence score greater than or equal to the given threshold.
        """
        filtered_predictions = Results()
        for prediction in self.predictions:
            if prediction.confidence is not None and prediction.confidence >= threshold:
                filtered_predictions.add_prediction(prediction)
        return filtered_predictions

    def filter_by_class_id(self, class_ids) -> 'Results':
        """
        Returns a new Results object containing only predictions
        with class_id matching one of the provided class_ids.
        
        Args:
            class_ids: Single class_id or list of class_ids to filter by
        """
        # Handle single class_id or list of class_ids
        if not isinstance(class_ids, (list, tuple)):
            class_ids = [class_ids]
            
        filtered_predictions = Results()
        for prediction in self.predictions:
            if prediction.class_id is not None and prediction.class_id in class_ids:
                filtered_predictions.add_prediction(prediction)
        return filtered_predictions

    def simplify(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the masks of all predictions in the Predictions object.

        Args:
            tolerance (float): The tolerance factor for simplification.
            preserve_topology (bool): Whether to preserve the topology.
        """
        for prediction in self.predictions:
            prediction.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def to_json(self):
        """
        Converts the list of predictions into a JSON string.
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)

    def to_dict(self):
        """
        Converts the list of predictions into a JSON string.
        """
        return [prediction.to_dict() for prediction in self.predictions]

    def to_json_with_metrics(self) -> str:
        """
        Converts the list of predictions into a JSON string.
        More to come here
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)


def from_datamarkin_api(api_response: dict, zones: Zones = None) -> Results:
    """
    Converts the Datamarkin API response to a `Predictions` object, filtering based on zones.

    Args:
        api_response (dict): The API response in dictionary format.
        zones (Zones, optional): The Zones object for managing included/excluded zones. If None, no filtering is applied.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    predictions_obj = Results()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Create the Prediction object
        prediction = Prediction(
            bbox=bbox,
            mask=mask,
            keypoints=keypoints,
            class_id=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        predictions_obj.add_prediction(prediction)

    return predictions_obj



def from_detectron2(detectron2_results) -> Results:
    """
    Converts Detectron2 results to a custom Results object.

    Args:
        detectron2_results: Detectron2 inference results containing instances with prediction data.

    Returns:
        Results: A unified Results object containing predictions.
    """
    predictions_obj = Results()
    
    # Get instances and ensure they're on CPU for processing
    instances = detectron2_results["instances"].to("cpu")
    
    # Check if we have any instances
    if len(instances) == 0:
        return predictions_obj

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
        
        # Create a Prediction object
        prediction = Prediction(
            bbox=bbox,
            masks=[mask] if mask is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            confidence=confidence
        )

        # Add the prediction to the Results object
        predictions_obj.add_prediction(prediction)

    return predictions_obj


def from_ultralytics(ultralytics_results) -> Results:
    """
    Converts Ultralytics YOLO results to a custom Results object.
    
    Supports both detection and segmentation models.
    
    Args:
        ultralytics_results: YOLO results from the Ultralytics library.

    Returns:
        Results: A unified Results object containing predictions.
    """
    predictions_obj = Results()
    
    # Handle empty results or single result
    if not ultralytics_results:
        return predictions_obj
        
    # Get the first result (YOLO returns a list with one result per image)
    result = ultralytics_results[0]
    
    # Handle case where there are no detections
    if result.boxes is None or len(result.boxes) == 0:
        return predictions_obj
    
    # Get all box data in one tensor transfer (more efficient)
    boxes_data = result.boxes.data.cpu().numpy()
    
    # Extract components from the tensor
    # Format: [x1, y1, x2, y2, conf, class_id, ...]
    xyxy = boxes_data[:, :4]  # Bounding boxes
    confidences = boxes_data[:, 4]  # Confidence scores  
    class_ids = boxes_data[:, 5].astype(int)  # Class IDs
    
    # Check if we have segmentation masks
    has_masks = hasattr(result, 'masks') and result.masks is not None
    
    # Process each detection
    num_detections = len(xyxy)
    for i in range(num_detections):
        # Basic detection info
        bbox = xyxy[i].tolist()
        confidence = float(confidences[i])
        class_id = int(class_ids[i])
        
        # Handle masks if available
        masks = None
        segments = None
        if has_masks:
            # Use polygon format (xy) for segments - it's more efficient
            segments = result.masks.xy[i]
            # Convert to integer coordinates
            if segments is not None and len(segments) > 0:
                segments = segments.astype(int).tolist()
                # Store segments as masks for compatibility
                # This maintains the expected interface without expensive pixel operations
                masks = [segments]  # Wrap in list as expected by Prediction
            
            # For pixel masks, we'll use the data attribute only when needed
            # This avoids expensive resizing operations unless absolutely necessary
            # masks.data is shape: [num_masks, height, width]
        
        # Create prediction object
        prediction = Prediction(
            bbox=bbox,
            masks=masks,  # Now properly populated with polygon segments
            segments=segments,
            keypoints=None,
            class_id=class_id,
            confidence=confidence
        )
        
        predictions_obj.add_prediction(prediction)
    
    # Store the original YOLO masks data for later use if needed
    # This avoids processing masks until they're actually used
    if has_masks:
        predictions_obj._ultralytics_masks = result.masks
    
    return predictions_obj


def from_transformers(transformers_results):
    pass


def from_sam(sam_results):
    pass


def from_datamarkin_csv(group, height, width) -> Results:
    """
    Converts CSV data to a `Predictions` object.

    Args:
        group: The pandas DataFrame group with the CSV rows.
        height: Image height to denormalize the bounding box and segmentation coordinates.
        width: Image width to denormalize the bounding box and segmentation coordinates.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    predictions_obj = Results()

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

        # Create the Prediction object
        prediction = Prediction(
            bbox=[xmin, ymin, xmax, ymax],
            mask=[segmentation_points],  # Add mask as list of lists of tuples
            keypoints=None,  # TODO
            class_id=row['class'],
            confidence=row.get('confidence', None)  # Add confidence if available
        )

        # Add the prediction to the predictions list
        predictions_obj.add_prediction(prediction)

    return predictions_obj

