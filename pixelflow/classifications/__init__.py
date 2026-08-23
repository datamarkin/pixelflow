# classifications/__init__.py
# Export the classification result type and its converters

from .classifications import Classification, Classifications

from .converters import (
    from_scores,
    from_ultralytics_classification,
)

__all__ = [
    'Classification',
    'Classifications',
    'from_scores',
    'from_ultralytics_classification',
]
