"""
Shared pytest fixtures for PixelFlow tests.

Provides reusable test data including sample images, detections, zones,
and mock objects for consistent testing across all test modules.
"""

import pytest
import numpy as np
import cv2
from typing import List, Tuple
import pixelflow as pf


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a blank white 640x480 RGB image."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image with colored rectangles for testing."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    # Add some colored rectangles using numpy (RGB format)
    image[100:200, 100:200] = (0, 0, 255)    # Blue
    image[150:250, 300:400] = (0, 255, 0)     # Green
    image[300:400, 450:550] = (255, 0, 0)     # Red
    return image


@pytest.fixture
def small_image() -> np.ndarray:
    """Create a small 100x100 RGB image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128


# ============================================================================
# Detection Fixtures
# ============================================================================

@pytest.fixture
def sample_keypoint() -> pf.detections.KeyPoint:
    """Create a sample keypoint."""
    return pf.detections.KeyPoint(x=100, y=200, name="nose", visibility=True)


@pytest.fixture
def sample_keypoints() -> List[pf.detections.KeyPoint]:
    """Create a list of sample keypoints for pose estimation."""
    return [
        pf.detections.KeyPoint(x=100, y=100, name="nose", visibility=True),
        pf.detections.KeyPoint(x=90, y=110, name="left_eye", visibility=True),
        pf.detections.KeyPoint(x=110, y=110, name="right_eye", visibility=True),
        pf.detections.KeyPoint(x=80, y=130, name="left_ear", visibility=False),
        pf.detections.KeyPoint(x=120, y=130, name="right_ear", visibility=True),
    ]


@pytest.fixture
def sample_detection() -> pf.detections.Detection:
    """Create a single sample detection."""
    return pf.detections.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        class_name="person",
        label="person 0.95"
    )


@pytest.fixture
def sample_detection_with_mask() -> pf.detections.Detection:
    """Create a detection with a binary mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return pf.detections.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.9,
        class_id=0,
        class_name="person",
        masks=[mask]
    )


@pytest.fixture
def sample_detection_with_keypoints(sample_keypoints) -> pf.detections.Detection:
    """Create a detection with keypoints."""
    return pf.detections.Detection(
        bbox=[50, 50, 150, 200],
        confidence=0.88,
        class_id=0,
        class_name="person",
        keypoints=sample_keypoints
    )


@pytest.fixture
def sample_detections() -> pf.detections.Detections:
    """Create a Detections container with multiple detections."""
    detections = pf.detections.Detections()

    # Add person detections
    detections.add_detection(pf.detections.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        class_name="person"
    ))

    detections.add_detection(pf.detections.Detection(
        bbox=[300, 150, 400, 280],
        confidence=0.87,
        class_id=0,
        class_name="person"
    ))

    # Add car detection
    detections.add_detection(pf.detections.Detection(
        bbox=[450, 300, 580, 420],
        confidence=0.92,
        class_id=2,
        class_name="car"
    ))

    # Add low confidence detection
    detections.add_detection(pf.detections.Detection(
        bbox=[50, 50, 100, 100],
        confidence=0.45,
        class_id=1,
        class_name="bicycle"
    ))

    return detections


@pytest.fixture
def empty_detections() -> pf.detections.Detections:
    """Create an empty Detections container."""
    return pf.detections.Detections()


# ============================================================================
# Zone Fixtures
# ============================================================================

@pytest.fixture
def sample_polygon() -> List[Tuple[int, int]]:
    """Create a sample polygon for zone testing."""
    return [(100, 100), (300, 100), (300, 300), (100, 300)]


@pytest.fixture
def sample_zone(sample_polygon) -> pf.zones.Zone:
    """Create a sample zone."""
    return pf.zones.Zone(
        polygon=sample_polygon,
        zone_id="zone_1",
        name="Test Zone",
        trigger_strategy="center"
    )


@pytest.fixture
def sample_zones() -> pf.Zones:
    """Create Zones container with multiple zones."""
    zones = pf.Zones()

    # Entrance zone
    zones.add_zone(
        polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
        zone_id="entrance",
        trigger_strategy="bottom_center"
    )

    # Exit zone
    zones.add_zone(
        polygon=[(400, 300), (600, 300), (600, 450), (400, 450)],
        zone_id="exit",
        trigger_strategy="center"
    )

    return zones


@pytest.fixture
def tracked_detections() -> pf.detections.Detections:
    """Create detections with tracking IDs."""
    detections = pf.detections.Detections()

    detections.add_detection(pf.detections.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        tracker_id=1,
        first_seen_time=0.0,
        tracking_duration=5.0
    ))

    detections.add_detection(pf.detections.Detection(
        bbox=[300, 150, 400, 280],
        confidence=0.87,
        class_id=0,
        tracker_id=2,
        first_seen_time=2.0,
        tracking_duration=3.0
    ))

    return detections


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_image_path(tmp_path, sample_image):
    """Create a temporary image file."""
    image_path = tmp_path / "test_image.jpg"
    # cv2.imwrite expects BGR, convert from RGB
    cv2.imwrite(str(image_path), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    return str(image_path)


@pytest.fixture
def temp_video_path(tmp_path, sample_image):
    """Create a temporary video file with 10 frames."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for i in range(10):
        # Add frame number to image
        frame = sample_image.copy()
        cv2.putText(frame, f"Frame {i}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv2.VideoWriter expects BGR, convert from RGB
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    return str(video_path)


