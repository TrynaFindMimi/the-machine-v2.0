from config.settings import VOLUME
from controllers.volume import VolumeController


def _vc() -> VolumeController:
    return VolumeController()


def test_mapping_near_to_far() -> None:
    vc = _vc()
    assert vc._vol_from_dist(VOLUME.near) == 0.0
    assert vc._vol_from_dist(VOLUME.far) == 1.0
    lo = vc._vol_from_dist(VOLUME.near + 0.01)
    hi = vc._vol_from_dist(VOLUME.far - 0.01)
    assert 0.0 < lo < hi < 1.0


def test_mapping_saturates() -> None:
    vc = _vc()
    assert vc._vol_from_dist(0.0) == 0.0
    assert vc._vol_from_dist(1.0) == 1.0


def test_feed_is_live_and_continuous() -> None:
    vc = _vc()
    lo, pending_lo = vc.feed(0.12)
    hi, pending_hi = vc.feed(0.42)
    assert 0.0 <= lo < hi <= 1.0
    assert pending_lo is not None
    assert pending_hi is not None
    assert vc.consume_pending_volume() == pending_hi


def test_volume_lowers_when_fingers_come_together() -> None:
    vc = _vc()
    vc.feed(0.90)
    open_vol = vc.feed(0.90)[0]
    for _ in range(50):
        vc.feed(0.05)
    closed_vol = vc.feed(0.05)[0]
    assert open_vol == 1.0
    assert closed_vol == 0.0
    assert closed_vol < open_vol


def test_no_feed_leaves_no_pending() -> None:
    vc = _vc()
    vc.feed(0.4)
    vc.consume_pending_volume()
    assert vc.consume_pending_volume() is None


def test_reset_clears_state() -> None:
    vc = _vc()
    vc.feed(0.3)
    vc.reset()
    assert vc.consume_pending_volume() is None
