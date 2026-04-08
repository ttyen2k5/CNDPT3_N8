"""Phan 5: UI Overlay - Hien thi thong tin hanh dong, FPS, trang thai."""

import cv2
import numpy as np
import config


class UIOverlay:
    """Ve overlay thong tin len frame camera - 2 hanh dong (Jump / Bend)."""

    ACTION_ICONS = {
        "Jump": "^^",
        "Bend": "v",
        "Idle": "-",
        "No Person": "?",
    }

    ACTION_COLORS = {
        "Jump": (150, 255, 150),
        "Bend": (100, 200, 255),
        "Idle": (180, 180, 180),
        "No Person": (0, 0, 255),
    }

    def __init__(self):
        self.fps_smooth = 0.0

    def draw(self, frame: np.ndarray, action: str,
             fps: float, tracking: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        result = frame.copy()

        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        self.fps_smooth = (config.FPS_SMOOTHING * self.fps_smooth
                           + (1 - config.FPS_SMOOTHING) * fps)

        cv2.putText(result, "NEON RUNNER - Pose Estimation", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_TITLE, 2)

        icon = self.ACTION_ICONS.get(action, "")
        action_color = self.ACTION_COLORS.get(action, config.COLOR_ACTION)
        cv2.putText(result, f"Action: {icon} {action}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_color, 2)

        cv2.putText(result, f"FPS: {int(self.fps_smooth)}", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_FPS, 2)

        status = "Tracking" if tracking else "No Person"
        status_color = config.COLOR_STATUS if tracking else (0, 0, 255)
        cv2.putText(result, status, (w - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        cv2.putText(result, "ESC=quit  D=debug", (w - 200, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        return result
