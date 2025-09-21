__version__ = "0.1.1"
__author__ = "Datamarkin"

# Import core modules
from . import detections

from . import media
from . import annotators as annotate
from . import colors
# from . import zones
# from . import crossings
from . import slicer
from . import smoother
from . import timer
from . import tracker

# Import specific functions for top-level access
from .media import Media, MediaInfo, write_frame, show_frame, close_display
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
    
    # Media handling
    "media",
    "Media",
    "MediaInfo", 
    "write_frame",
    "show_frame",
    "close_display",
    
    # Spatial analysis
    "zones",
    "Zones",
    "crossings", 
    "Crossings",
    
    # Processing utilities
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
