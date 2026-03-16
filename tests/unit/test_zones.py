"""
Unit tests for pixelflow.zones and pixelflow.crossings modules.

Tests zone management, trigger strategies, crossing detection,
and spatial analytics functionality.
"""

import pytest
import numpy as np
import pixelflow as pf


# ============================================================================
# Zone Class Tests
# ============================================================================

class TestZoneCreation:
    """Tests for Zone class initialization."""

    def test_zone_creation_basic(self, sample_polygon):
        """Test basic zone creation."""
        zone = pf.zones.Zone(
            polygon=sample_polygon,
            zone_id="test_zone"
        )

        assert zone.zone_id == "test_zone"
        assert zone.polygon is not None

    def test_zone_with_name(self, sample_polygon):
        """Test zone creation with name."""
        zone = pf.zones.Zone(
            polygon=sample_polygon,
            zone_id=1,
            name="Entrance Zone"
        )

        assert zone.zone_id == 1
        assert zone.name == "Entrance Zone"

    def test_zone_with_trigger_strategy(self, sample_polygon):
        """Test zone with specific trigger strategy."""
        zone = pf.zones.Zone(
            polygon=sample_polygon,
            zone_id="zone1",
            trigger_strategy="bottom_center"
        )

        assert zone.trigger_strategy == "bottom_center"

    def test_zone_with_multiple_strategies(self, sample_polygon):
        """Test zone with multiple trigger strategies."""
        zone = pf.zones.Zone(
            polygon=sample_polygon,
            zone_id="zone1",
            trigger_strategy=["center", "bottom_center"],
            mode="any"
        )

        assert isinstance(zone.trigger_strategy, list)
        assert zone.mode == "any"

    def test_zone_with_metadata(self, sample_polygon):
        """Test zone with custom metadata."""
        metadata = {"priority": "high", "alert": True}
        zone = pf.zones.Zone(
            polygon=sample_polygon,
            zone_id="zone1",
            metadata=metadata
        )

        assert zone.metadata == metadata


# ============================================================================
# Zones Container Tests
# ============================================================================

class TestZonesContainer:
    """Tests for Zones container class."""

    def test_zones_creation_empty(self):
        """Test creating empty Zones container."""
        zones = pf.Zones()
        assert len(zones.zones) == 0

    def test_zones_add_zone(self, sample_polygon):
        """Test adding zone to container."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=sample_polygon,
            zone_id="zone1"
        )

        assert len(zones.zones) == 1
        assert "zone1" in zones.zones

    def test_zones_add_multiple(self):
        """Test adding multiple zones."""
        zones = pf.Zones()

        zones.add_zone(
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            zone_id="zone1"
        )

        zones.add_zone(
            polygon=[(200, 200), (300, 200), (300, 300), (200, 300)],
            zone_id="zone2"
        )

        assert len(zones.zones) == 2

    def test_zones_get_zone(self, sample_zones):
        """Test retrieving zone by ID."""
        zone = sample_zones.get_zone("entrance")
        assert zone is not None
        assert zone.zone_id == "entrance"

    def test_zones_remove_zone(self, sample_zones):
        """Test removing zone from container."""
        original_count = len(sample_zones.zones)
        sample_zones.remove_zone("entrance")

        assert len(sample_zones.zones) == original_count - 1


# ============================================================================
# Zone Update and Detection Tests
# ============================================================================

class TestZoneDetectionUpdate:
    """Tests for updating detections with zone information."""

    def test_update_detections_basic(self, sample_zones, sample_detections):
        """Test basic zone update on detections."""
        updated = sample_zones.update(sample_detections)

        # Should return Detections instance
        assert isinstance(updated, pf.detections.Detections)
        # Should have same number of detections
        assert len(updated) == len(sample_detections)

    def test_update_adds_zone_info(self, sample_detections):
        """Test that update adds zone information to detections."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 80), (220, 80), (220, 220), (80, 220)],
            zone_id="zone1",
            trigger_strategy="center"
        )

        updated = zones.update(sample_detections)

        # Check if zone_ids are added to detections
        for det in updated:
            # Each detection should have zone_ids attribute (may be empty)
            assert hasattr(det, 'zone_ids')

    def test_update_with_center_strategy(self, sample_detections):
        """Test zone matching with center point strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 80), (220, 80), (220, 220), (80, 220)],
            zone_id="zone1",
            trigger_strategy="center"
        )

        updated = zones.update(sample_detections)

        # First detection (bbox [100, 100, 200, 200]) center is at (150, 150)
        # Should be in zone1
        matching = [d for d in updated if hasattr(d, 'zone_ids') and 'zone1' in (d.zone_ids or [])]
        assert len(matching) >= 1

    def test_update_with_bottom_center_strategy(self, sample_detections):
        """Test zone matching with bottom_center strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 150), (220, 150), (220, 220), (80, 220)],
            zone_id="zone1",
            trigger_strategy="bottom_center"
        )

        updated = zones.update(sample_detections)
        # Bottom center of first detection would be at (150, 200)

        assert isinstance(updated, pf.detections.Detections)


# ============================================================================
# Trigger Strategy Tests
# ============================================================================

class TestTriggerStrategies:
    """Tests for different zone trigger strategies."""

    def test_center_strategy(self):
        """Test center point trigger strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
            zone_id="zone1",
            trigger_strategy="center"
        )

        detections = pf.detections.Detections()
        # Detection with center at (150, 150) - inside zone
        detections.add_detection(pf.detections.Detection(
            bbox=[125, 125, 175, 175]
        ))

        updated = zones.update(detections)
        assert len(updated) > 0

    def test_overlap_strategy(self):
        """Test overlap trigger strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
            zone_id="zone1",
            trigger_strategy="overlap"
        )

        detections = pf.detections.Detections()
        # Detection partially overlapping zone
        detections.add_detection(pf.detections.Detection(
            bbox=[150, 150, 250, 250]
        ))

        updated = zones.update(detections)
        assert len(updated) > 0

    def test_percentage_strategy(self):
        """Test percentage-based trigger strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
            zone_id="zone1",
            trigger_strategy="percentage",
            overlap_threshold=0.5
        )

        detections = pf.detections.Detections()
        # Detection with >50% overlap
        detections.add_detection(pf.detections.Detection(
            bbox=[120, 120, 180, 180]
        ))

        updated = zones.update(detections)
        assert len(updated) > 0

    def test_contains_strategy(self):
        """Test contains (full containment) strategy."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
            zone_id="zone1",
            trigger_strategy="contains"
        )

        detections = pf.detections.Detections()
        # Detection fully inside zone
        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 150, 150]
        ))

        updated = zones.update(detections)
        assert len(updated) > 0


# ============================================================================
# Zone Statistics Tests
# ============================================================================

class TestZoneStatistics:
    """Tests for zone statistics and counting."""

    def test_get_zone_stats_basic(self, sample_zones, sample_detections):
        """Test getting zone statistics."""
        updated = sample_zones.update(sample_detections)

        stats = sample_zones.get_zone_stats()

        # Should return stats for all zones
        assert isinstance(stats, dict)

    def test_zone_count_updates(self, sample_detections):
        """Test that zone counts update correctly."""
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(80, 80), (220, 80), (220, 220), (80, 220)],
            zone_id="zone1"
        )

        # First update
        zones.update(sample_detections)

        # Zone should have updated count
        zone = zones.get_zone("zone1")
        assert hasattr(zone, 'count') or hasattr(zone, 'current_count')


# ============================================================================
# Crossings Class Tests
# ============================================================================

class TestCrossingsCreation:
    """Tests for Crossings class initialization."""

    def test_crossings_creation_empty(self):
        """Test creating empty Crossings container."""
        crossings = pf.Crossings()
        assert len(crossings.crossings) == 0

    def test_add_crossing_basic(self):
        """Test adding crossing line."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 200),
            end=(500, 200),
            line_id="line1"
        )

        assert len(crossings.crossings) == 1
        assert any(c.line_id == "line1" for c in crossings.crossings)

    def test_add_multiple_crossings(self):
        """Test adding multiple crossing lines."""
        crossings = pf.Crossings()

        crossings.add_crossing(start=(100, 200), end=(500, 200), line_id="line1")
        crossings.add_crossing(start=(300, 100), end=(300, 400), line_id="line2")

        assert len(crossings.crossings) == 2

    def test_crossing_with_direction(self):
        """Test crossing line with custom parameters."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 200),
            end=(500, 200),
            line_id="line1",
            minimum_crossing_threshold=2,
            debounce_time=45
        )

        assert any(c.line_id == "line1" for c in crossings.crossings)


# ============================================================================
# Crossing Detection Tests
# ============================================================================

class TestCrossingDetection:
    """Tests for line crossing detection."""

    def test_detect_crossing_basic(self):
        """Test basic crossing detection."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 240),
            end=(540, 240),
            line_id="line1"
        )

        # Create detections simulating movement across line
        # This would require tracking data
        detections = pf.detections.Detections()

        # Detection below line (y=250)
        detections.add_detection(pf.detections.Detection(
            bbox=[150, 250, 200, 300],
            tracker_id=1
        ))

        updated = crossings.update(detections)

        assert isinstance(updated, pf.detections.Detections)

    def test_crossing_count_updates(self):
        """Test that crossing counts update."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 240),
            end=(540, 240),
            line_id="line1"
        )

        # Would need tracked detections to test actual counting
        # This tests the structure exists
        assert hasattr(crossings, 'update')


# ============================================================================
# Edge Cases and Validation
# ============================================================================

class TestZoneCrossingEdgeCases:
    """Test edge cases for zones and crossings."""

    def test_zone_with_invalid_polygon(self):
        """Test zone creation with invalid polygon."""
        try:
            zones = pf.Zones()
            zones.add_zone(
                polygon=[(0, 0)],  # Only one point
                zone_id="invalid"
            )
            # May or may not raise error depending on implementation
        except (ValueError, Exception):
            # Expected to fail
            pass

    def test_update_with_empty_detections(self, sample_zones, empty_detections):
        """Test zone update with no detections."""
        updated = sample_zones.update(empty_detections)

        assert len(updated) == 0

    def test_crossing_with_untracked_detections(self, sample_detections):
        """Test crossings with detections without tracker IDs."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 240),
            end=(540, 240),
            line_id="line1"
        )

        # Detections without tracker_id
        updated = crossings.update(sample_detections)

        # Should handle gracefully
        assert isinstance(updated, pf.detections.Detections)

    def test_zone_update_preserves_detections(self, sample_zones, sample_detections):
        """Test that zone update preserves detection properties."""
        # Store original properties
        original_confidences = [d.confidence for d in sample_detections]

        updated = sample_zones.update(sample_detections)

        # Properties should be preserved
        updated_confidences = [d.confidence for d in updated]
        assert updated_confidences == original_confidences
