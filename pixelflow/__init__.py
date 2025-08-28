__version__ = "0.1.0"
__author__ = "Datamarkin"

# Import core modules
from . import draw
from . import results
from . import video
from . import annotators as annotate
from . import colors
from . import zones
from . import lines
from . import slicer
from . import buffer

# Import specific functions for top-level access
from .video import lazy_frame_generator
from .zones import Zones, Zone
from .lines import Lines, Line
from .slicer import SlicedInference, auto_slice_size
from .buffer import Buffer

# Define the public API
__all__ = ["draw", "annotate", "results", "slicer", "buffer", 
           "lazy_frame_generator", "Zones", "Zone", "Lines", "Line", 
           "SlicedInference", "auto_slice_size", "Buffer",
           "__version__", "__author__"]
