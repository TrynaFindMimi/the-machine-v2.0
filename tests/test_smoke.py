import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def test_core_imports() -> None:
    import app.registry  # noqa: F401
    import app.vision  # noqa: F401
    import common.fps  # noqa: F401
    import config.palette  # noqa: F401
    import config.settings  # noqa: F401
    import config.strings  # noqa: F401
    import controllers.gestures  # noqa: F401
    import controllers.hand  # noqa: F401
    import controllers.thumb  # noqa: F401
    import controllers.volume  # noqa: F401
    import core.fingers  # noqa: F401
    import core.perceptron  # noqa: F401
    import core.playlist  # noqa: F401
    import infrastructure.capture  # noqa: F401
    import infrastructure.display  # noqa: F401
    import infrastructure.player  # noqa: F401
    import presentation.modes.hand  # noqa: F401
    import presentation.modes.line  # noqa: F401
    import presentation.modes.music  # noqa: F401
    import presentation.modes.position  # noqa: F401

    assert config.settings.CAMERA_INDEX >= 0
    assert app.registry.TESTS


def test_playlist_scans_sagas() -> None:
    from core.playlist import scan_sagas

    sagas = scan_sagas()
    assert sagas
    songs = [str(s) for saga in sagas for s in saga]
    assert songs


def test_player_degrades_without_audio() -> None:
    import infrastructure.player as player_mod

    player = player_mod.MusicPlayer()
    player.audio_ok = False
    player.volume = 1.0
    player.play()
    player.pause()
    player.resume()
    player.stop()
    player.tick()
    player.set_volume(0.5)
    assert not player.playing


def test_play_does_not_advance_when_already_playing() -> None:
    import infrastructure.player as player_mod

    player = player_mod.MusicPlayer()
    player.audio_ok = True
    player.volume = 1.0
    player.play()
    first = player.song_idx
    player.play()
    player.play()
    assert player.song_idx == first
    assert player.playing
    player.stop()


def test_sidebar_glued_to_right_edge() -> None:
    import numpy as np

    from config.settings import WINDOW
    from presentation.ui.layout import draw_sidebar
    from presentation.ui.theme import SIDEBAR_W

    canvas = np.zeros((WINDOW.height, WINDOW.width, 3), dtype=np.uint8)
    before = canvas.copy()
    draw_sidebar(canvas, "hand", 1, 30.0)

    assert np.any(canvas[:, WINDOW.width - SIDEBAR_W :] != 0)
    sep = WINDOW.width - SIDEBAR_W
    assert np.array_equal(canvas[:, : sep - 2], before[:, : sep - 2])
