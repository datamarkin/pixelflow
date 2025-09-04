__version__ = "0.1.0"
__author__ = "Datamarkin"

# Import core modules
from . import detections
from . import video
from . import annotators as annotate
from . import colors
from . import zones
from . import crossings
from . import slicer
from . import buffer
from . import timer
from . import tracker

# Import specific functions for top-level access
from .video import get_video_frames, VideoInfo, VideoWriter, Monitor
from .zones import Zones
from .crossings import Crossings
from .slicer import SlicedInference, auto_slice_size
from .buffer import Buffer
from .timer import TimeTracker

# Define the public API
__all__ = ["annotate", "detections", "slicer", "buffer", "timer", "video", "tracker",
           "get_video_frames", "VideoInfo", "VideoWriter", "Monitor",
           "Zones", "Crossings", "TimeTracker",
           "SlicedInference", "auto_slice_size", "Buffer",
           "__version__", "__author__"]
