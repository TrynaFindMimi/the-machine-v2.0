# Capas y módulos

| Capa | Directorio | Módulos | Imports permitidos | Descripción |
|------|------------|---------|--------------------|-------------|
| Config | `config/` | `palette.py`, `strings.py`, `settings.py` | nada | Constantes BGR, textos UI, `MODES_ORDER`, `AudioSettings` |
| Transversal | `common/` | `fps.py` | stdlib | `FPSCounter` EMA |
| Dominio | `core/` | `perceptron.py`, `results.py`, `handedness.py`, `gestures.py`, `fingers.py`, `playlist.py` | `config`, `common`, `numpy` | Lógica pura sin cv2/pygame. `handedness.py` aísla corrección flip `get_handedness`/`normalize_handedness`; `results.py` solo `get_gesture`/`to_pixel_points`/`count_hands`; `fingers.py` features por dedo (indice/medio/anular/meñique por coseno del angulo de flexion `MCP→PIP→TIP` con `EXTEND_THRESHOLDS`, pulgar por `thumb_palm_features` + Perceptron con fallback geométrico `_thumb_extended`) + `index_thumb_distance`/`pinky_thumb_distance`/`hand_scale`; `playlist.py` `scan_sagas` |
| Controladores | `controllers/` | `hand.py`, `thumb.py`, `gestures.py`, `volume.py` | `config`, `core`, `numpy` | `FingerRNN` x4 + `HandController` (count con mayoría de pulgar / action / train / save / load); `ThumbPerceptron` (3 features de palma, `models/thumb.npz`); `MusicGestureController` (ventana, agreement, settle 0.45s, rate limit 1.5s, sin re-petición, acción pendiente); `VolumeController` (Perceptron 2-f, mapeo vivo indice↔pulgar distancia→0..1, sin commit extra) |
| Presentación UI | `presentation/ui/` | `theme.py`, `drawing.py`, `effects.py`, `layout.py`, `greek.py` | `config`, `core` | Dibujo de la UI: `layout.draw_sidebar` (panel BASALT + friso + MODE/HANDS/FPS/CONTROLS), motivos griegos en `greek.py`, `effects.apply_cctv_effect` como no-op |
| Presentación Modes | `presentation/modes/` | `hand.py`, `line.py`, `position.py`, `music.py` | `controllers`, `core`, `config`, `common`, `presentation/ui` | Un `draw()` por modo. `music.py` NO importa `infrastructure`; expone `set_context`/`consume_pending_action`/`consume_pending_volume` |
| Aplicación | `app/` | `registry.py`, `vision.py`, `runner.py` | todos los anteriores + `infrastructure` | Orquesta captura→visión→modo→UI; crea `MusicPlayer`, inyecta contexto music, despacha acciones |
| Infraestructura | `infrastructure/` | `capture.py`, `display.py`, `player.py` | `config`, `core/playlist` | Adapters cv2/pygame. `player.py` zquets `MusicPlayer` sobre `pygame.mixer` con `AUDIO` de settings |
| Fachada | `main.py` | — | `app`, `config` | Solo parsea args |

Añadir nuevo modo: crear `presentation/modes/nuevo.py` con `draw()`, registrar en `app/registry.py`, añadir título en `config/strings.py`.

Regla de capas para el modo music: `presentation/modes/music.py` no conoce `infrastructure` — la decisión gesto→acción vive en `controllers/gestures.MusicGestureController` (`.pending`) y en `controllers/volume.VolumeController` (`.pending`), y el runner (`app/runner.py`) las consume (`consume_pending_action`/`consume_pending_volume`) y aplica sobre `MusicPlayer`. `infrastructure/player.py` lee el dispositivo de audio de `config/settings.AUDIO`.
