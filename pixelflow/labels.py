"""Class name resolution, shared by every result type.

A model emits class *ids*; the names for those ids are metadata that has to come
from somewhere - the checkpoint, or the caller. Detections and Classifications
both need that lookup, and the three formats a caller may supply it in are the
same either way, so the rule lives here rather than once per package. Where a
name cannot be found the id is carried and the name stays None: a name is never
invented.
"""

__all__ = ["get_label_info", "build_name_index"]


def get_label_info(labels, class_id):
    """Extract class name and keypoint names from various label formats.

    Supports:
        List[str]      — ["person", "car"] — index = class_id
        Dict[int, str] — {0: "person", 1: "car"} — key = class_id
        List[dict]     — [{"id": 0, "name": "person", "keypoints": [...]}] — search by "id" field

    Returns:
        tuple: (class_name, kp_names) where kp_names is List[str] or None.
    """
    if labels is None or class_id is None:
        return None, None

    # Rich label format: List[dict] with "id"/"name" keys
    if isinstance(labels, list) and labels and isinstance(labels[0], dict):
        for label in labels:
            if label["id"] == class_id:
                name = label["name"]
                kp_names = [kp["name"] for kp in label.get("keypoints", [])] or None
                return name, kp_names
        return None, None

    # Dict format: {int: str}
    if isinstance(labels, dict):
        return labels.get(class_id), None

    # Simple list format: ["person", "car"]
    if isinstance(labels, list) and class_id < len(labels):
        return labels[class_id], None

    return None, None


def build_name_index(labels):
    """Return an ``{id: name}`` mapping for `labels`, or None when there is nothing to map.

    `get_label_info` answers for one id at a time, which is right where a converter
    handles a handful of instances. A classifier scores every class it knows - 1000 for
    an ImageNet head - and the ``List[dict]`` format scans linearly per lookup, so asking
    it once per row is quadratic. Building the map once is not.

    Accepts the same three formats and resolves to the same answers, None included.
    """
    if labels is None:
        return None

    # Rich label format: List[dict] with "id"/"name" keys
    if isinstance(labels, list) and labels and isinstance(labels[0], dict):
        return {label["id"]: label["name"] for label in labels}

    # Dict format is already the index.
    if isinstance(labels, dict):
        return labels

    # Simple list format: position is the id
    if isinstance(labels, list):
        return dict(enumerate(labels))

    return None
