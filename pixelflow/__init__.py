__version__ = "0.1.3"
__author__ = "Datamarkin"

# Import core modules
from . import detections

from . import media
from . import annotators as annotate
from . import transforms as transform
from . import colors
# from . import zones
# from . import crossings
from . import slicer
from . import smoother
from . import timer
from . import tracker
from . import assets
from .classes import COCO_LABELS

# Import specific functions for top-level access
from .media import (
    DisplayExit, VideoReader, CameraStream, VideoWriter,
    read_image, read_video, display_video, display_image, save_image,
    close_display, to_pil, from_pil,
)
from .zones import Zones
from .crossings import Crossings
from .slicer import SlicedInference, auto_slice_size
from .buffer import Buffer
from .smoother import smooth
from .timer import TimeTracker

# Define the public API
__all__ = [
    # Core data structures
    "detections",

    # Visual components
    "annotate",
    "colors",

    # Image transformations
    "transform",

    # Media handling
    "media",
    "DisplayExit",
    "VideoReader",
    "CameraStream",
    "VideoWriter",
    "read_image",
    "read_video",
    "display_video",
    "display_image",
    "save_image",
    "close_display",
    "to_pil",
    "from_pil",

    # Spatial analysis
    "zones",
    "Zones",
    "crossings",
    "Crossings",

    # Processing utilities
    "assets",
    "slicer",
    "SlicedInference",
    "auto_slice_size",
    "Buffer",
    "smooth",

    # Performance & tracking
    "timer",
    "TimeTracker",
    "tracker",

    # Constants
    "COCO_LABELS",

    # Metadata
    "__version__",
    "__author__"
]
