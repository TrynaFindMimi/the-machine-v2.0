# Modos de presentacion (presentation/modes/)

Cada modo expone `draw(frame, results) -> (frame, hand_count)`.

## hand (`presentation/modes/hand.py`)

Skeleton fino `1px` + landmarks `2px` numerados `0.3` + bounding box viewfinder blanco + label handedness corregido por flip. Crosshair si `hand_count==0`.

## line (`presentation/modes/line.py`)

Perceptron en vivo por mano (P_A=4 thumb tip, P_B=8 index tip). `build_dataset` + `Perceptron.train_budget(budget=200)` cada frame. Linea `BLUE` `1px` entre P5→P9. `FINGERS_TOGETHER_THRESH=0.04` resetea perceptrón. Soporta 2 manos, `_perc[0]/[1]` persistentes.

## position (`presentation/modes/position.py`)

`GestureRecognizer` con `num_hands=2` (ambas manos). Paleta predominante `BLACK`/`WHITE` (sin `GESTURE_COLORS`): `draw_bbox`/`draw_skeleton`/`draw_landmarks` todo en `WHITE 2px` sin borde negro. Tipografía grande y legible con caja negra: título `0.70/2`, gesto `0.65/2` + `Left/Right` `0.60/2` + HUD `0.60/2` vía `_put_text_box()` (rect `BLACK` + texto `WHITE`). Muestra `gesture + confianza %` por mano (máx 2) y usa `core/handedness.get_handedness` (flip corregido) + `core/results.get_gesture`.

## music (`presentation/modes/music.py`)

Reproductor de EPIC: The Musical por sagas. Procesa ambas manos: derecha para navegacion/accion, izquierda para volumen (`core/handedness.get_handedness` + flip corregido). UI de tags negros estilo position vía `_put_text_box`: título `0.70/2`, SAGA `0.55/2`, SONG `0.45/1`, TRACK `0.45/1`, STATE `0.55/1`, VOLUME `0.55/1` (volumen actual del reproductor, junto al STATE), FINGERS `0.5/1`, ACTION `0.5/1`, etiquetas Right/Left junto a sus bboxes, barra de volumen con label VOLUME bajo la mano izquierda.

Pipeline por frame (mano derecha — acciones):
- `core/fingers.features_from_landmarks` → 5 features binarias (±1) por dedo. Para índice/medio/anular/meñique: coseno del ángulo de flexión entre segmentos `MCP→PIP` y `PIP→TIP` (`_finger_extension`), independiente de la orientación de la mano; umbrales `EXTEND_THRESHOLDS=(0.45, 0.45, 0.35, 0.45)`. Para el pulgar: **Perceptrón de 3 features** (`controllers/thumb.py`) — `[min_dist(P4→{5,7,9,13,17})/hand_scale, dentro_cuadrilátero(5,17,1,9), bias]` — entrenado con polos sintéticos "doblado sobre la palma / contiguo al índice" (−1) vs "extendido lejos" (+1). + 6ª feature hand-sign (no usada para contar).
- `controllers/hand.HandController.count(window 8x6)` → pulgar por **mayoría de la ventana** (`window[:,0] > 0`) + 4 `FingerRNN` (índice/medio/anular/meñique) → suma de dedos detectados.
- `controllers/gestures.MusicGestureController.feed(feat)` → ventana `len=8`, agreement `>=4` frames, settle `>=0.45s` sin cambios, rate limit `1.5s` entre acciones, mapeo count→accion.
- `consume_pending_action()` → el runner despacha `player.play()/pause()/prev/next song|saga`.

Mano izquierda — volumen:
- `core/fingers.index_thumb_distance` → distancia euclidea index↔pulgar tip↔tip normalizada por `hand_scale`.
- `controllers/volume.VolumeController.feed(dist)` → `Perceptron` (2 features, entrenado cerca/lejos) mapea distancia a nivel continuo 0..1 (index+pulgar juntos=0, separados=100) y lo deja en `.pending` **en vivo, cada frame**: `consume_pending_volume()` → `player.set_volume()` sin necesidad de gesto extra. La izquierda controla el volumen entero con index+pulgar.

Las acciones de la mano derecha llevan **settle** anti-sweep (`MusicGestureController`, `SETTLE_SECONDS=0.45`): la accion solo se despacha cuando el conteo lleva esa fraccion de segundo sin cambiar, y cada gesto dispara una sola vez (no se repite mientras se mantiene, `_last_fired`); además hay rate limit de `1.5s` entre acciones.

Gestos (mano derecha): 0=PREV SAGA, 1=PREV SONG, 2=PAUSE, 3=PLAY, 4=NEXT SONG, 5=NEXT SAGA. Pulgar por **Perceptrón** (`controllers/thumb.py`, `models/thumb.npz`) y 4 `FingerRNN` (2 capas ocultas, hidden=8, dropout 0.3 en cada capa; `models/finger_1..4.npz`) con **early-stop por accuracy** (`HandController.train_all`): entrena hasta tener `acc` y `val_acc` > 0.85 con diferencia ≤ 0.002; `Perceptron` (2 features, 3000 epochs) en `models/volume.npz`; si faltan, se reentrenan al iniciar.

La UI del modo music **persiste** la info de reproduccion: los tags SAGA, SONG, TRACK, STATE y VOLUME se dibujan en cada frame aunque no haya mano en camara (solo FINGERS/ACTION pasan a desconocido cuando no hay mano). El STATE refleja PLAY/PAUSE y el VOLUME el nivel real del reproductor (`player.volume`), que se pasa via `set_context`.

**Zona efectiva**: rectángulo central donde se reconocen los comandos (volumen con la mano izquierda y acciones con la derecha). Distancia entre sus bordes y los de la pantalla: `MUSIC_EFFECTIVE_ZONE_FRACTION` (1/8) del alto y del ancho del frame. Se dibuja con `_effective_zone(w, h)`; el comando solo actúa si el **centro del bbox** de la mano (`_hand_center`) cae dentro (`_point_in_zone`). El recuadro cambia a `GREEN` cuando al menos una mano relevante está dentro y `GRAY` cuando no; fuera de la zona la mano derecha resetea el detector de gestos y la izquierda no alimenta el volumen.

## Ciclo

`hand → line → position → music → hand` con `n`. Inicio: `python main.py [hand|line|position|music]`. Registry en `app/registry.py`.
