# PixelFlow Test Suite

Comprehensive test suite for the PixelFlow computer vision library.

## Overview

- **Total Tests**: 263
- **Current Pass Rate**: 143 passing (54%)
- **Test Framework**: pytest
- **Coverage Tool**: pytest-cov
- **Structure**: Unit tests + Integration tests

## Quick Start

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=pixelflow --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_detections.py

# Run tests matching pattern
pytest -k "test_filter"
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures (images, detections, zones)
├── unit/                          # Unit tests (isolated components)
│   ├── test_detections.py         # Detection, Detections, KeyPoint classes
│   ├── test_filters.py            # All filter methods (45 tests)
│   ├── test_converters.py         # Framework converters (Ultralytics, OCR, etc.)
│   ├── test_annotators.py         # Visual annotators (box, label, mask, etc.)
│   ├── test_transforms.py         # Image & detection transforms
│   ├── test_zones.py              # Zone management and crossings
│   └── test_utilities.py          # Media, Buffer, Timer, etc.
├── integration/                   # Integration tests (workflows)
│   ├── test_end_to_end.py         # Complete pipelines
│   └── test_method_chaining.py    # Filter chaining, serialization
└── fixtures/                      # Test data (images, videos, mock data)
```

## Test Categories

### Unit Tests (187 tests)

#### `test_detections.py` (25 tests)
Tests for core detection data structures:
- ✅ KeyPoint creation and serialization
- ✅ Detection creation (basic, with masks, with keypoints)
- ✅ Detections container (add, iterate, index)
- ✅ JSON serialization
- ✅ Binary mask base64 encoding/decoding
- ⚠️ Some properties need API alignment

#### `test_filters.py` (68 tests)
Tests for all detection filter methods:
- ✅ Confidence filtering
- ✅ Class ID filtering
- ✅ Size and dimension filters
- ✅ Aspect ratio filtering
- ✅ Method chaining
- ⚠️ Some filter signatures need adjustment
- ⚠️ OCR filters pending Detection class updates
- ⚠️ Tracking filters pending tracking field support

#### `test_converters.py` (18 tests)
Tests for framework-specific converters:
- ✅ Mock Ultralytics converter
- ✅ Mock Detectron2 converter
- ⚠️ OCR converters pending Detection OCR field support
- ⚠️ SAM converter not yet implemented
- ⚠️ Datamarkin API converter needs signature check

#### `test_annotators.py` (42 tests)
Tests for visualization functions:
- ✅ Box annotator
- ✅ Label annotator
- ✅ Mask annotator
- ✅ Blur (privacy) annotator
- ✅ Pixelate (privacy) annotator
- ✅ Polygon annotator
- ✅ Oval annotator
- ✅ Multi-layer annotation
- ⚠️ Some tests fail due to Detection constructor differences

#### `test_transforms.py` (36 tests)
Tests for image and detection transforms:
- ✅ Image rotation, flipping, cropping
- ✅ Enhancement (CLAHE, grayscale, gamma)
- ✅ Detection-aware rotation
- ✅ Detection-aware flipping
- ✅ Detection-aware cropping
- ⚠️ Some parameter names need adjustment
- ⚠️ Keypoint alignment API differs

#### `test_zones.py` (27 tests)
Tests for spatial analytics:
- ✅ Zone creation and management
- ✅ Zones container
- ✅ Trigger strategies (center, overlap, etc.)
- ⚠️ Zone update return type differs
- ⚠️ Crossings API differs from expected

#### `test_utilities.py` (41 tests)
Tests for utility modules:
- ✅ Media loading and iteration
- ✅ MediaInfo metadata
- ✅ Colors palettes
- ✅ Validators
- ⚠️ Buffer API differs
- ⚠️ SlicedInference API differs
- ⚠️ Smoother API differs
- ⚠️ TimeTracker API differs

### Integration Tests (76 tests)

#### `test_end_to_end.py` (42 tests)
Complete workflow tests:
- ✅ Detection pipeline (filter → annotate)
- ✅ Privacy protection pipeline
- ✅ Multi-class processing
- ✅ Zone analytics workflows
- ✅ Transform pipelines
- ✅ Video frame processing
- ✅ Performance tests

#### `test_method_chaining.py` (34 tests)
Method chaining and serialization:
- ✅ Simple filter chains
- ✅ Complex multi-filter chains
- ✅ Chain immutability
- ✅ JSON serialization roundtrip
- ⚠️ OCR chains pending OCR field support
- ⚠️ Tracking chains pending tracking fields

## Fixtures

### Image Fixtures
- `blank_image`: White 640x480 BGR image
- `sample_image`: Image with colored rectangles
- `small_image`: 100x100 test image

### Detection Fixtures
- `sample_keypoint`: Single KeyPoint
- `sample_keypoints`: List of 5 keypoints
- `sample_detection`: Basic detection
- `sample_detection_with_mask`: Detection with binary mask
- `sample_detection_with_keypoints`: Detection with keypoints
- `sample_detections`: Container with 4 detections
- `empty_detections`: Empty container
- `sample_ocr_detection`: Detection with OCR data
- `tracked_detections`: Detections with tracking IDs

### Zone Fixtures
- `sample_polygon`: Rectangular polygon
- `sample_zone`: Single zone
- `sample_zones`: Container with 2 zones

### File Fixtures
- `temp_image_path`: Temporary JPG file
- `temp_video_path`: Temporary MP4 with 10 frames

## Test Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Slow-running tests (>1 second)
@pytest.mark.requires_model # Tests requiring ML model downloads
```

Usage:
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

## Coverage

Generate coverage report:
```bash
# Terminal report
pytest --cov=pixelflow --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=pixelflow --cov-report=html
open htmlcov/index.html
```

Current coverage targets:
- **Immediate goal**: 70% coverage
- **Long-term goal**: 85%+ coverage

## Common Test Patterns

### Testing Detection Filters
```python
def test_my_filter(self, sample_detections):
    """Test custom filter."""
    filtered = sample_detections.my_filter(param=value)

    # Verify results
    assert len(filtered) > 0
    for det in filtered:
        assert det.meets_criteria()

    # Verify immutability
    assert len(sample_detections) == 4  # Original unchanged
```

### Testing Annotators
```python
def test_my_annotator(self, blank_image, sample_detections):
    """Test custom annotator."""
    annotated = pf.annotate.my_annotator(blank_image.copy(), sample_detections)

    assert annotated.shape == blank_image.shape
    assert not np.array_equal(annotated, blank_image)  # Modified
```

### Testing Transforms
```python
def test_my_transform(self, sample_image, sample_detections):
    """Test custom transform."""
    transformed_img, transformed_dets = pf.transform.my_transform(
        sample_image,
        sample_detections,
        param=value
    )

    assert transformed_img.shape[2] == 3  # Still BGR
    assert len(transformed_dets) == len(sample_detections)
```

## Extending Tests

### Adding New Test File

1. Create file in appropriate directory:
```bash
touch tests/unit/test_new_module.py
```

2. Import required modules:
```python
import pytest
import numpy as np
import pixelflow as pf
```

3. Organize into test classes:
```python
class TestMyFeature:
    """Tests for my feature."""

    def test_basic_functionality(self, sample_detections):
        """Test basic usage."""
        result = pf.my_feature(sample_detections)
        assert result is not None
```

### Adding New Fixtures

Add to `tests/conftest.py`:
```python
@pytest.fixture
def my_fixture():
    """Description of fixture."""
    # Setup
    data = create_test_data()

    yield data

    # Teardown (optional)
    cleanup(data)
```

## Known Issues & How to Fix

### API Alignment Needed

Some tests fail due to differences between expected API and actual implementation. **All issues are documented with fixes:**

1. **Detection Constructor**: Some fields like `label`, `text`, `tracking_duration` not in constructor
2. **Filter Signatures**: Parameter names differ (e.g., `min_confidence` vs `confidence`)
3. **Buffer API**: Constructor uses `frames` not `size`
4. **SlicedInference API**: Different initialization parameters
5. **Crossings API**: Method is `add_crossing` not `add_line`


**Target:** 85%+ pass rate (220+ tests) achievable in ~6-8 hours of work.

## Best Practices

1. **Use Fixtures**: Always use shared fixtures from `conftest.py`
2. **Test Immutability**: Verify operations don't modify original data
3. **Test Edge Cases**: Include empty, None, and boundary values
4. **Clear Names**: Use descriptive test function names
5. **Single Assertion Focus**: Each test should verify one thing
6. **Arrange-Act-Assert**: Structure tests clearly
7. **Use Markers**: Mark slow or special tests appropriately

## Running CI Tests

The test suite is designed to run in CI/CD pipelines:

```yaml
# .github/workflows/tests.yml (example)
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest --cov=pixelflow --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Debugging Failed Tests

```bash
# Show full stack trace
pytest -vv --tb=long

# Stop on first failure
pytest -x

# Run specific test
pytest tests/unit/test_detections.py::TestDetection::test_detection_creation_basic

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb
```

## Performance Testing

For performance-critical code:
```python
@pytest.mark.slow
def test_performance(self, benchmark):
    """Test performance of operation."""
    def operation():
        # Code to benchmark
        pass

    result = benchmark(operation)
```

## Contributing

When adding new features to PixelFlow:

1. **Write tests first** (TDD approach recommended)
2. **Ensure tests pass** before submitting PR
3. **Add docstrings** to all test functions
4. **Update this README** if adding new test categories
5. **Maintain >70% coverage** on new code

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [PixelFlow Documentation](https://pixelflow.datamarkin.com)
- [Test-Driven Development Guide](https://martinfowler.com/bliki/TestDrivenDevelopment.html)
