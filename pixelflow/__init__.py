__version__ = "0.3.0"
__author__ = "Datamarkin"

# Import core modules
from . import media
from . import annotators as annotate
from . import transforms as transform
from . import colors
from . import slicer
from . import smoother
from . import timer
from . import tracker
from . import assets

# Result types and their converters. Every model output PixelFlow understands becomes
# one of two things: Detections for anything that localises, Classifications for
# anything that only names. They are peers, not variants of each other.
#
# Converters are flat -- pf.from_ultralytics(...) -- and each subpackage owns its own
# export list, so a new one is declared once rather than re-typed here. Detection is
# the unmarked case because almost every model that produces a container localises
# something; only classification needs the suffix.
from .detections import *
from .detections import __all__ as _DETECTION_EXPORTS
from .classifications import *
from .classifications import __all__ as _CLASSIFICATION_EXPORTS

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
    # Result types and converters, declared by the subpackages that own them
    *_DETECTION_EXPORTS,
    *_CLASSIFICATION_EXPORTS,

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

    # Metadata
    "__version__",
    "__author__"
]
