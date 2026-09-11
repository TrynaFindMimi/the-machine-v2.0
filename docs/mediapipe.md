# Uso de MediaPipe

## HandLandmarker (hand, line) — `app/vision.make_landmarker()`

```python
HandLandmarkerOptions(
  base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
  running_mode=VIDEO, num_hands=20, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
)
results = landmarker.detect_for_video(mp_img, ts)
# results.hand_landmarks, results.handedness
```

## GestureRecognizer (position, 2 manos) — `app/vision.make_recognizer()`

```python
GestureRecognizerOptions(
  base_options=BaseOptions(model_asset_path="models/gesture_recognizer.task"),
  running_mode=VIDEO, num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
)
results = recognizer.recognize_for_video(mp_img, ts)
# + results.gestures (None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou)
```

Preproceso: `cv2.flip` + `resize` en `infrastructure/capture.Camera.read()`, BGR→RGB antes de `mp.Image`.

Landmarks 21 ptos: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky. P_A=4, P_B=8 para perceptron. Helpers: `core/results.py` → `to_pixel_points`, `get_gesture`, `count_hands`; `core/handedness.py` → `get_handedness`/`normalize_handedness` (corrige flip espejo `Left↔Right`).
