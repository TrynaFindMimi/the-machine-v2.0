from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.drawing_styles import get_default_hand_connections_style
from mediapipe.tasks.python.vision.drawing_utils import _CONNECTION, DrawingSpec, draw_landmarks

try:
    from config.settings import VISION

    _MODEL_PATH = VISION.landmarker_path
except Exception:
    _MODEL_PATH = "hand_landmarker.task"
conn_style = get_default_hand_connections_style()
HAND_CONNECTIONS = [_CONNECTION(a, b) for a, b in conn_style.keys()]

white = (255, 255, 255)
landmark_spec = DrawingSpec(color=white, thickness=1, circle_radius=2)
connection_spec = DrawingSpec(color=white, thickness=1, circle_radius=1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=6,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = hand_landmarker.detect_for_video(mp_img, frame_timestamp)
    frame_timestamp += 1

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(
                frame,
                hand_landmarks,
                HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec,
            )
    cv2.imshow("hand_recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
