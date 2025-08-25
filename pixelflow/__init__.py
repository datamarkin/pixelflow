__version__ = "0.1.0"
__author__ = "Datamarkin"

# Import core modules
from . import draw
from . import results
from . import video
from . import annotate
from . import colors
from . import zones
from . import lines

# Import specific functions for top-level access
from .video import lazy_frame_generator
from .zones import Zones, Zone
from .lines import Lines, Line

# Define the public API
__all__ = ["draw", "annotate", "results", "zones", "lines", "lazy_frame_generator",
           "Zones", "Zone", "Lines", "Line", "__version__", "__author__"]
