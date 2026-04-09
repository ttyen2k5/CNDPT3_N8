"""Phần 2: Pose Estimation - Nhận diện khung xương bằng MediaPipe Tasks API."""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]


def _ensure_model():
    """Tải model pose_landmarker nếu chưa có."""
    if os.path.exists(_MODEL_PATH):
        return
    print("[INFO] Downloading pose_landmarker_lite.task ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[OK] Model downloaded successfully.")


class PoseEstimator:
    """Sử dụng MediaPipe PoseLandmarker (Tasks API) để nhận diện keypoints."""

    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    def __init__(self):
        _ensure_model()

        with open(_MODEL_PATH, "rb") as f:
            model_data = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_idx = 0

    def process(self, frame):
        """
        Xử lý frame để nhận diện pose.
        Trả về (keypoints_dict, annotated_frame).
        keypoints_dict: {name: (x, y, z, visibility)} tọa độ chuẩn hóa [0,1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_idx += 1
        timestamp_ms = self._frame_idx * 33  # ~30 FPS

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = frame.copy()
        keypoints = {}

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

            self._draw_landmarks(annotated, landmarks)

            for idx, name in enumerate(self.LANDMARK_NAMES):
                lm = landmarks[idx]
                keypoints[name] = (lm.x, lm.y, lm.z, lm.visibility)

        return keypoints, annotated

    @staticmethod
    def _draw_landmarks(frame, landmarks):
        """Vẽ khung xương lên frame."""
        h, w = frame.shape[:2]
        pts = {}
        for i, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            pts[i] = (px, py)
            if lm.visibility > 0.5:
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                if landmarks[a].visibility > 0.5 and landmarks[b].visibility > 0.5:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)

    def release(self):
        self.landmarker.close()
