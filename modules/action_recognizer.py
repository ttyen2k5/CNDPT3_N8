"""Phan 3: Action Recognition - su dung model ML (Jump/Bend)."""

from collections import deque
from pathlib import Path

import joblib
import numpy as np

import config


class ActionRecognizer:
    """Predict action from model.pkl and map to game labels."""

    LANDMARK_ORDER = [
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
        payload = joblib.load(Path(config.MODEL_PATH))
        self.model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        self.frame_features = deque(maxlen=config.MODEL_FEATURE_WINDOW)
        self.pred_window = deque(maxlen=config.MODEL_FEATURE_WINDOW)
        self.current_action = "Idle"
        self._last_confidence = 0.0
        self._last_raw_pred = None
        self._labels = {0: "Bend", 1: "Jump"}

    def recognize(self, keypoints):
        if not keypoints:
            self.current_action = "No Person"
            return self.current_action

        frame_feat = self._build_frame_feature(keypoints)
        self.frame_features.append(frame_feat)

        if len(self.frame_features) < config.MODEL_MIN_READY_FRAMES:
            self.current_action = "Idle"
            return self.current_action

        agg = self._build_aggregated_feature().reshape(1, -1)
        pred = int(self.model.predict(agg)[0])
        probs = self.model.predict_proba(agg)[0]

        self._last_raw_pred = pred
        self._last_confidence = float(probs[pred])

        if self._last_confidence < config.MODEL_CONFIDENCE_THRESHOLD:
            self.pred_window.append(-1)
        else:
            self.pred_window.append(pred)

        votes = list(self.pred_window)
        smooth_pred = max(set(votes), key=votes.count)
        self.current_action = self._labels.get(smooth_pred, "Idle")
        return self.current_action

    def _build_frame_feature(self, keypoints):
        feat = np.zeros((33, 3), dtype=np.float32)
        for i, name in enumerate(self.LANDMARK_ORDER):
            pt = keypoints.get(name)
            if pt:
                feat[i] = [pt[0], pt[1], pt[2]]
        return feat.reshape(-1)

    def _build_aggregated_feature(self):
        arr = np.stack(self.frame_features, axis=0)
        mean_feat = arr.mean(axis=0)
        std_feat = arr.std(axis=0)
        min_feat = arr.min(axis=0)
        max_feat = arr.max(axis=0)
        feat = np.concatenate([mean_feat, std_feat, min_feat, max_feat], axis=0)
        mu = feat.mean()
        sigma = feat.std() + 1e-8
        feat = (feat - mu) / sigma
        return feat.astype(np.float32)

    def get_debug_info(self, keypoints):
        if not keypoints:
            return {}
        return {
            "model": "RandomForest",
            "raw_pred": self._labels.get(self._last_raw_pred, "N/A"),
            "conf": f"{self._last_confidence:.3f} (thr={config.MODEL_CONFIDENCE_THRESHOLD})",
            "window": f"{len(self.pred_window)}/{config.MODEL_FEATURE_WINDOW}",
        }
