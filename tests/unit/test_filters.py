"""
Unit tests for pixelflow.detections.filters module.

Tests all filter methods attached to the Detections class including
confidence, class, size, position, zone, and tracking filters.
"""

import pytest
import numpy as np
import pixelflow as pf


# ============================================================================
# Basic Filter Tests
# ============================================================================

class TestConfidenceFilter:
    """Tests for filter_by_confidence method."""

    def test_filter_by_confidence_basic(self, sample_detections):
        """Test basic confidence filtering."""
        filtered = sample_detections.filter_by_confidence(0.8)
        assert len(filtered) == 3  # 0.95, 0.87, 0.92 should pass
        for det in filtered:
            assert det.confidence >= 0.8

    def test_filter_by_confidence_high_threshold(self, sample_detections):
        """Test with high confidence threshold."""
        filtered = sample_detections.filter_by_confidence(0.93)
        assert len(filtered) == 1  # Only 0.95
        assert filtered[0].confidence == 0.95

    def test_filter_by_confidence_zero_threshold(self, sample_detections):
        """Test with zero threshold (should return all)."""
        filtered = sample_detections.filter_by_confidence(0.0)
        assert len(filtered) == 4

    def test_filter_by_confidence_empty_result(self, sample_detections):
        """Test filtering that results in empty detections."""
        filtered = sample_detections.filter_by_confidence(0.99)
        assert len(filtered) == 0

    def test_filter_by_confidence_returns_new_instance(self, sample_detections):
        """Test that filter returns new Detections instance."""
        filtered = sample_detections.filter_by_confidence(0.8)
        assert filtered is not sample_detections
        assert len(sample_detections) == 4  # Original unchanged


class TestClassFilter:
    """Tests for filter_by_class_id method."""

    def test_filter_by_single_class(self, sample_detections):
        """Test filtering by single class ID."""
        filtered = sample_detections.filter_by_class_id(0)  # person
        assert len(filtered) == 2
        for det in filtered:
            assert det.class_id == 0

    def test_filter_by_multiple_classes(self, sample_detections):
        """Test filtering by multiple class IDs."""
        filtered = sample_detections.filter_by_class_id([0, 2])  # person, car
        assert len(filtered) == 3
        class_ids = [det.class_id for det in filtered]
        assert 0 in class_ids
        assert 2 in class_ids
        assert 1 not in class_ids

    def test_filter_by_nonexistent_class(self, sample_detections):
        """Test filtering by non-existent class ID."""
        filtered = sample_detections.filter_by_class_id(999)
        assert len(filtered) == 0


class TestRemapClassIds:
    """Tests for remap_class_ids method."""

    def test_remap_class_ids_basic(self, sample_detections):
        """Consolidate several source class IDs onto a single target ID."""
        remapped = sample_detections.remap_class_ids([0, 1, 2], 99)

        assert all(det.class_id == 99 for det in remapped)

    def test_remap_class_ids_partial_mapping(self, sample_detections):
        """IDs outside from_ids are left untouched."""
        remapped = sample_detections.remap_class_ids(0, 100)

        person_count = sum(1 for d in remapped if d.class_id == 100)
        assert person_count == 2
        # Other classes should remain unchanged
        assert any(d.class_id == 1 for d in remapped)
        assert any(d.class_id == 2 for d in remapped)

    def test_remap_class_ids_accepts_scalar_from_id(self, sample_detections):
        """A bare int is accepted as from_ids, not just a list."""
        remapped = sample_detections.remap_class_ids(2, 42)

        assert sum(1 for d in remapped if d.class_id == 42) == 1


# ============================================================================
# Size and Dimension Filters
# ============================================================================

class TestSizeFilters:
    """Tests for size-based filtering methods."""

    def test_filter_by_size_min_area(self, sample_detections):
        """Test filtering by minimum area."""
        # Person 1: 100x100 = 10000
        # Person 2: 100x130 = 13000
        # Car: 130x120 = 15600
        # Bicycle: 50x50 = 2500
        filtered = sample_detections.filter_by_size(min_area=11000)
        assert len(filtered) == 2  # Person 2 and Car

    def test_filter_by_size_max_area(self, sample_detections):
        """Test filtering by maximum area."""
        filtered = sample_detections.filter_by_size(max_area=12000)
        assert len(filtered) == 2  # Person 1 and Bicycle

    def test_filter_by_size_range(self, sample_detections):
        """Test filtering by area range."""
        filtered = sample_detections.filter_by_size(min_area=5000, max_area=14000)
        assert len(filtered) == 2  # Person 1 and Person 2

    def test_filter_by_dimensions_min_width(self, sample_detections):
        """Test filtering by minimum width."""
        filtered = sample_detections.filter_by_dimensions(min_width=120)
        # Car has width 130, Person 2 has width 100
        assert len(filtered) == 1

    def test_filter_by_dimensions_max_height(self, sample_detections):
        """Test filtering by maximum height."""
        filtered = sample_detections.filter_by_dimensions(max_height=110)
        # Person 1: height=100, Bicycle: height=50
        assert len(filtered) == 2

    def test_filter_by_aspect_ratio(self, sample_detections):
        """Test filtering by aspect ratio."""
        # Looking for roughly square detections (aspect ratio close to 1.0)
        filtered = sample_detections.filter_by_aspect_ratio(
            min_ratio=0.9, max_ratio=1.1
        )
        # Person 1 is 100x100 (ratio=1.0)
        # Bicycle is 50x50 (ratio=1.0)
        assert len(filtered) >= 2


class TestRelativeSizeFilter:
    """Tests for filter_by_relative_size method."""

    def test_filter_by_relative_size(self, sample_detections):
        """Test filtering by size relative to total frame area."""
        # Frame is 640x480 = 307200 px. min_percent is a *fraction* of frame
        # area despite the name, so 0.02 means "at least 2% of the frame".
        filtered = sample_detections.filter_by_relative_size(
            min_percent=0.02,
            frame_width=640,
            frame_height=480,
        )
        # person 10000px (3.3%), person 13000px (4.2%), car 15600px (5.1%) pass;
        # the 50x50 bicycle at 2500px (0.8%) is dropped.
        assert len(filtered) == 3
        assert all(d.class_name != "bicycle" for d in filtered)


# ============================================================================
# Position Filters
# ============================================================================

class TestPositionFilter:
    """Tests for filter_by_position method."""

    def test_filter_by_position_left_side(self, sample_detections):
        """The "left" region keeps boxes whose centre falls inside the margin."""
        # margin_x = 640 * 0.3 = 192; person(150) and bicycle(75) qualify.
        filtered = sample_detections.filter_by_position(
            "left", margin_percent=0.3, frame_width=640, frame_height=480
        )
        assert len(filtered) == 2
        assert {d.class_name for d in filtered} == {"person", "bicycle"}

    def test_filter_by_position_top(self, sample_detections):
        """The "top" region keeps boxes above the vertical margin."""
        # margin_y = 480 * 0.4 = 192; person(150) and bicycle(75) qualify.
        filtered = sample_detections.filter_by_position(
            "top", margin_percent=0.4, frame_width=640, frame_height=480
        )
        assert len(filtered) == 2

    def test_filter_by_position_center(self, sample_detections):
        """The "center" region excludes the margin band on every side."""
        filtered = sample_detections.filter_by_position(
            "center", margin_percent=0.1, frame_width=640, frame_height=480
        )
        # All four sample boxes sit inside the 64/48px inset.
        assert len(filtered) == 4

    def test_filter_by_position_requires_frame_dims(self, sample_detections):
        """Frame dimensions are mandatory — omitting them is an error."""
        with pytest.raises(ValueError):
            sample_detections.filter_by_position("center")


# ============================================================================
# Zone Filters
# ============================================================================

class TestZoneFilter:
    """Tests for filter_by_zones method."""

    def test_filter_by_zones_single(self, sample_detections):
        """Test filtering by single zone."""
        # First update detections with zone info
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 80), (220, 80), (220, 220), (80, 220)],
            zone_id="zone1"
        )
        updated = zones.update(sample_detections)

        # Filter for zone1
        filtered = updated.filter_by_zones("zone1")
        # Person 1 should be in this zone
        assert len(filtered) >= 1

    def test_filter_by_zones_multiple(self, sample_detections):
        """Test filtering by multiple zones."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 80), (220, 80), (220, 220), (80, 220)],
            zone_id="zone1"
        )
        zones.add_zone(
            polygon=[(280, 130), (420, 130), (420, 300), (280, 300)],
            zone_id="zone2"
        )
        updated = zones.update(sample_detections)

        filtered = updated.filter_by_zones(["zone1", "zone2"])
        assert len(filtered) >= 2


# ============================================================================
# Tracking Filters
# ============================================================================

class TestTrackingFilters:
    """Tests for tracking-related filter methods."""

    def test_filter_tracked_objects(self, tracked_detections):
        """Test filtering only tracked objects."""
        filtered = tracked_detections.filter_tracked_objects()
        assert len(filtered) == 2
        for det in filtered:
            assert det.tracker_id is not None

    def test_filter_by_tracking_duration(self, tracked_detections):
        """Test filtering by tracking duration."""
        # Filter for objects tracked for at least 4 seconds
        filtered = tracked_detections.filter_by_tracking_duration(min_seconds=4.0)
        assert len(filtered) == 1
        assert filtered[0].total_time == 5.0

    def test_filter_by_first_seen_time(self, tracked_detections):
        """Test filtering by first seen time."""
        # Filter for objects first seen after time 1.0
        filtered = tracked_detections.filter_by_first_seen_time(start_time=1.0)
        assert len(filtered) == 1
        assert filtered[0].first_seen_time == 2.0


# ============================================================================
# Duplicate and Overlap Filters
# ============================================================================

class TestDuplicateFilters:
    """Tests for duplicate and overlap filtering."""

    def test_remove_duplicates_basic(self):
        """Test removing duplicate detections."""
        detections = pf.detections.Detections()

        # Add overlapping detections of same class
        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            class_id=0
        ))
        detections.add_detection(pf.detections.Detection(
            bbox=[105, 105, 205, 205],  # Slightly overlapping
            confidence=0.85,
            class_id=0
        ))

        filtered = detections.remove_duplicates(iou_threshold=0.5)
        # Should keep only the higher confidence detection
        assert len(filtered) <= 2

    def test_filter_overlapping_basic(self):
        """Test filtering overlapping detections."""
        detections = pf.detections.Detections()

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            class_id=0
        ))
        detections.add_detection(pf.detections.Detection(
            bbox=[150, 150, 250, 250],  # Overlaps with first
            class_id=0
        ))

        filtered = detections.filter_overlapping(min_overlap=0.3)
        # Should remove overlapping detections
        assert len(filtered) <= 2


# ============================================================================
# Method Chaining Tests
# ============================================================================

class TestMethodChaining:
    """Tests for filter method chaining."""

    def test_basic_chaining(self, sample_detections):
        """Test chaining multiple filters."""
        filtered = (sample_detections
                   .filter_by_confidence(0.7)
                   .filter_by_class_id(0))

        # Should have only high-confidence person detections
        assert len(filtered) == 2
        for det in filtered:
            assert det.confidence >= 0.7
            assert det.class_id == 0

    def test_complex_chaining(self, sample_detections):
        """Test complex filter chain."""
        filtered = (sample_detections
                   .filter_by_confidence(0.8)
                   .filter_by_size(min_area=8000)
                   .filter_by_class_id([0, 2]))

        # Each filter should be applied in sequence
        assert len(filtered) >= 0
        for det in filtered:
            assert det.confidence >= 0.8
            assert det.class_id in [0, 2]
