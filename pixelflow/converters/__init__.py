"""
PixelFlow Converters Module

Internal module containing conversion functions for different ML frameworks.
These functions convert framework-specific detection results to PixelFlow's
unified Detections format.

Note: This module is for internal use. Import converters from pixelflow.detections instead:
    from pixelflow.detections import from_detectron2, from_ultralytics, etc.
"""