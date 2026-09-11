from __future__ import annotations

from presentation.modes.music import (
    _effective_zone,
    _hand_center,
    _point_in_zone,
)


def test_zone_margins_are_one_eighth() -> None:
    zone = _effective_zone(1280, 720)
    assert zone == (160, 90, 1120, 630)


def test_zone_center_is_inside() -> None:
    zone = _effective_zone(1280, 720)
    assert _point_in_zone((640, 360), zone)


def test_zone_excludes_screen_edges() -> None:
    zone = _effective_zone(1280, 720)
    assert not _point_in_zone((0, 360), zone)
    assert not _point_in_zone((640, 0), zone)
    assert not _point_in_zone((1279, 360), zone)
    assert not _point_in_zone((640, 719), zone)


def test_zone_boundary_is_inclusive() -> None:
    zone = _effective_zone(1280, 720)
    assert _point_in_zone((160, 90), zone)
    assert _point_in_zone((1120, 630), zone)


def test_hand_center_is_bbox_midpoint() -> None:
    assert _hand_center([(10, 20), (30, 50)]) == (20, 35)
