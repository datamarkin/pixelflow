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
def sample_keypoint() -> pf.KeyPoint:
    """Create a sample keypoint."""
    return pf.KeyPoint(x=100, y=200, id=0, name="nose", confidence=0.9)


@pytest.fixture
def sample_keypoints() -> List[pf.KeyPoint]:
    """Create a list of sample keypoints for pose estimation."""
    return [
        pf.KeyPoint(x=100, y=100, id=0, name="nose", confidence=0.9),
        pf.KeyPoint(x=90, y=110, id=1, name="left_eye", confidence=0.9),
        pf.KeyPoint(x=110, y=110, id=2, name="right_eye", confidence=0.9),
        pf.KeyPoint(x=80, y=130, id=3, name="left_ear", confidence=0.0),
        pf.KeyPoint(x=120, y=130, id=4, name="right_ear", confidence=0.9),
    ]


@pytest.fixture
def sample_detection() -> pf.Detection:
    """Create a single sample detection."""
    return pf.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        class_name="person"
    )


@pytest.fixture
def sample_detection_with_mask() -> pf.Detection:
    """Create a detection with a binary mask.

    The mask is full-frame (480x640, matching blank_image/sample_image) because
    annotators require mask dimensions to match the frame they draw onto.
    """
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True  # matches the bbox below
    return pf.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.9,
        class_id=0,
        class_name="person",
        masks=[mask]
    )


@pytest.fixture
def sample_detection_with_keypoints(sample_keypoints) -> pf.Detection:
    """Create a detection with keypoints."""
    return pf.Detection(
        bbox=[50, 50, 150, 200],
        confidence=0.88,
        class_id=0,
        class_name="person",
        keypoints=sample_keypoints
    )


@pytest.fixture
def sample_detections() -> pf.Detections:
    """Create a Detections container with multiple detections."""
    detections = pf.Detections()

    # Add person detections
    detections.add_detection(pf.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        class_name="person"
    ))

    detections.add_detection(pf.Detection(
        bbox=[300, 150, 400, 280],
        confidence=0.87,
        class_id=0,
        class_name="person"
    ))

    # Add car detection
    detections.add_detection(pf.Detection(
        bbox=[450, 300, 580, 420],
        confidence=0.92,
        class_id=2,
        class_name="car"
    ))

    # Add low confidence detection
    detections.add_detection(pf.Detection(
        bbox=[50, 50, 100, 100],
        confidence=0.45,
        class_id=1,
        class_name="bicycle"
    ))

    return detections


@pytest.fixture
def empty_detections() -> pf.Detections:
    """Create an empty Detections container."""
    return pf.Detections()


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


# ============================================================================
# Tracking Fixtures
# ============================================================================

@pytest.fixture
def tracked_detections() -> pf.Detections:
    """Create detections with tracking IDs."""
    detections = pf.Detections()

    detections.add_detection(pf.Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        class_id=0,
        tracker_id=1,
        first_seen_time=0.0,
        total_time=5.0
    ))

    detections.add_detection(pf.Detection(
        bbox=[300, 150, 400, 280],
        confidence=0.87,
        class_id=0,
        tracker_id=2,
        first_seen_time=2.0,
        total_time=3.0
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


def _write_video(path, frames, fps=30.0, codec="mp4v"):
    """Write RGB frames to a video file and return the path."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps,
                             (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return str(path)


@pytest.fixture
def portrait_video_path(tmp_path):
    """A 480x640 portrait video.

    The suite was landscape-only, which is why a reader reporting a transposed size
    could never have failed a test. Portrait is the shape that catches it.
    """
    frames = []
    for i in range(10):
        frame = np.full((640, 480, 3), 40, dtype=np.uint8)
        cv2.putText(frame, str(i), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 3)
        frames.append(frame)
    return _write_video(tmp_path / "portrait.mp4", frames, fps=25.0)


@pytest.fixture
def counted_video_path(tmp_path):
    """A 25 fps, 20-frame video whose frames are individually identifiable.

    Each frame is a solid shade of its own index, so a test can assert *which*
    frames came back -- which is what seeking and striding need in order to be
    checked at all.
    """
    frames = [np.full((120, 160, 3), i * 10, dtype=np.uint8) for i in range(20)]
    return _write_video(tmp_path / "counted.mp4", frames, fps=25.0)


@pytest.fixture
def exif_rotated_image_path(tmp_path):
    """A 200x100 JPEG tagged Orientation=6, i.e. displayed rotated to 100x200."""
    from PIL import Image

    array = np.zeros((100, 200, 3), dtype=np.uint8)
    array[:, :100] = 255
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 CW for display
    path = tmp_path / "rotated.jpg"
    image.save(str(path), exif=exif)
    return str(path)


@pytest.fixture
def rgba_image_path(tmp_path):
    """A PNG with an alpha channel."""
    path = tmp_path / "rgba.png"
    rgba = np.zeros((40, 60, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    cv2.imwrite(str(path), rgba)
    return str(path)


@pytest.fixture
def grayscale_image_path(tmp_path):
    """A single-channel PNG."""
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), np.full((40, 60), 90, dtype=np.uint8))
    return str(path)


@pytest.fixture
def temp_video_path(tmp_path, sample_image):
    """Create a temporary video file with 10 frames."""
    frames = []
    for i in range(10):
        frame = sample_image.copy()
        cv2.putText(frame, f"Frame {i}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frames.append(frame)
    return _write_video(tmp_path / "test_video.mp4", frames)


