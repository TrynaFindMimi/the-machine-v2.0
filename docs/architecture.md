# Arquitectura del proyecto

## Capas

```
main.py              → fachada mínima (args → app.runner.run)
app/                 → orquestación: runner + vision + registry
presentation/        → UI: modes/ (hand/line/position/music) + ui/ (theme/layout/drawing/effects)
controllers/         → gesto→accion: hand.py (FingerRNN x4, HandController), thumb.py (ThumbPerceptron), gestures.py (MusicGestureController), volume.py (VolumeController, Perceptron)
core/                → dominio puro: fingers, perceptron, results, gestures, handedness, playlist (sin cv2/pygame)
config/              → configuración: palette, strings, settings (hojas)
common/              → transversal: fps
infrastructure/      → adapters: capture (cv2.VideoCapture), display (pygame), player (pygame.mixer)
models/              → modelos .task + finger_1..4.npz (RNN por dedo) + thumb.npz + volume.npz (Perceptron)
music/               → sagas mp3 (EPIC: The Musical)
```

### Flujo de datos

```
Webcam → infrastructure/capture.Camera.read() [flip+resize 1060x720]
    ↓
app/vision.py → HandLandmarker (hand/line/music) | GestureRecognizer (position, 2 manos)
    ↓
presentation/modes/*.draw(frame, results) → (frame, hand_count)  [bbox viewfinder + skeleton/landmarks 2px + crosshair, WHITE sobre caja BLACK]
    ↓
    presentation/ui/greek + presentation/ui/layout.draw_sidebar (panel BASALT con friso y MODE/HANDS/FPS/CONTROLS)
    ↓
infrastructure/display.Window.show(canvas 1280x720) [BGR→RGB→pygame]
```

> `presentation/ui/effects.apply_cctv_effect` queda como no-op (el antiguo efecto CCTV de scanlines/viñeta está desactivado en `app/runner.py`); los efectos decorativos reales viven en `presentation/ui/greek.py` (`draw_panel`, `draw_fret_band`, `draw_pediment`, `draw_laurel_divider`, `draw_text_centered`).

Flujo del modo music (por frame, bifurcacion sobre el anterior):

```
[mano derecha] presentation/modes/music.draw(right hand landmarks)
    ↓ core/fingers.features_from_landmarks → 5 binarias (±1): indice/medio/anular/meñique por coseno del angulo de flexion MCP→PIP→TIP (umbrales 0.45 / 0.45 / 0.35 / 0.45); pulgar por Perceptron 3-features (dist P4→{5,7,9,13,17} / within quad (5,17,1,9) / bias) + hand-sign
    ↓ controllers/gestures.MusicGestureController.feed → window(8), agreement(4), settle(0.45s), rate limit(1.5s)
    ↓ controllers/hand.HandController.count → pulgar por mayoría de ventana + 4x FingerRNN → count → action(0=PREV SAGA,1=PREV SONG,2=PAUSE,3=PLAY,4=NEXT SONG,5=NEXT SAGA)
    ↓ .pending queda como accion
app/runner.consume_pending_action() → player.play()/pause()/prev|next song|saga

[mano izquierda] presentation/modes/music.draw(left hand landmarks)
    ↓ core/fingers.index_thumb_distance / pinky_thumb_distance (normalizadas por hand_scale)
    ↓ controllers/volume.VolumeController.feed → Perceptron (2 features) → nivel 0..1; commit al unir meñique↔pulgar, re-arm al separar
    ↓ .pending queda como volumen
app/runner.consume_pending_volume() → player.set_volume()
    ↓ pygame.mixer.music (devicename desde config/settings.AUDIO) → sink PulseAudio/bluetooth
```

### Dependencias entre capas

```
config/palette.py, config/strings.py, config/settings.py, common/fps.py     ← hojas
core/fingers.py → config (nada) | core/perceptron.py → (nada) | core/results.py → (nada) | core/handedness.py → (nada) | core/gestures.py → config/palette | core/playlist.py → (nada)
controllers/hand.py → config/strings + numpy | controllers/thumb.py → config/settings + core/fingers + core/perceptron | controllers/gestures.py → controllers/hand + controllers/thumb | controllers/volume.py → core/perceptron + config/settings
presentation/ui/theme.py → (nada) | presentation/ui/drawing.py → config/palette | presentation/ui/effects.py → config/palette
presentation/ui/layout.py → config/palette + config/strings + presentation/ui/theme + presentation/ui/greek  # draw_sidebar (panel BASALT + friso) + draw_fret_band
presentation/modes/*.py → controllers/* + core/* + config/* + common/fps + presentation/ui/*   # music → controllers/gestures + controllers/volume + core/fingers
app/registry.py → presentation/modes/* | app/vision.py → mediapipe | app/runner.py → app/* + infrastructure + presentation/ui/* + config/strings
infrastructure/capture.py → cv2 | infrastructure/display.py → pygame+cv2 | infrastructure/player.py → pygame + config/settings + core/playlist
main.py → app/registry + app/runner + config/strings
```

Reglas: `core` nunca importa `presentation`; `config/common` nunca importan capas superiores; `presentation` no importa `app/infrastructure`. El modo music comunica acciones y volumen con el runner a traves de `MusicGestureController.pending` y `VolumeController.pending` (`consume_pending_action`/`consume_pending_volume`), sin que `presentation` toque pygame. `core/handedness.py` aísla la corrección de flip (Left↔Right) para que `core/results.py` solo haga conversión geométrica/gestos. UI temática griega con sidebar BASALT (sin header/footer).

Para el informe tecnico completo (objetivos, tecnicas de ML, deteccion de dedos, fine-tuning de umbrales, limitaciones y trabajo futuro) ver [INFO.md](informe.md).

### Estilo visual

Temática griega (EPIC: The Musical). Paleta en `config/palette.py` (`IVORY`, `STONE`, `BASALT`, `BRONZE`, `GOLD`, `OLIVE`, `CLAY`, `MARBLE`). `presentation/ui/layout.draw_sidebar` pinta un panel `BASALT` con `draw_fret_band` (friso), separadores `BRONZE` y texto con acentos `GOLD`/`IVORY`/`STONE`; `OLIVE` indica HANDS>0. `presentation/ui/greek.py` reutiliza motivos (friso, frontón, laurel, texto centrado). El esqueleto y bbox de cada modo se dibujan en `WHITE` (`BLACK` para las cajas de texto `_put_text_box`), crosshair si no hay mano. `apply_cctv_effect` existe como no-op (efecto CCTV desactivado). La tipografía es `FONT_HERSHEY_SIMPLEX` (0.3-0.7), con `0.60-0.70` en position/music sobre caja `BLACK` para legibilidad.
