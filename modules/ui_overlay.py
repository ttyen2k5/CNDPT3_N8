"""Phan 5: UI Overlay - Hien thi thong tin hanh dong, FPS, trang thai."""

import cv2
import numpy as np
import config


class UIOverlay:
    """Ve overlay thong tin len frame camera - 6 hanh dong."""

    ACTION_ICONS = {
        "Raise Hand": "^",
        "Wave": "~",
        "Running": ">>",
        "Walking": ">",
        "Jump": "^^",
        "Bend": "v",
        "Idle": "-",
        "No Person": "?",
    }

    ACTION_COLORS = {
        "Raise Hand": (0, 255, 255),
        "Wave": (255, 100, 255),
        "Running": (100, 200, 255),
        "Walking": (180, 220, 180),
        "Jump": (150, 255, 150),
        "Bend": (100, 255, 100),
        "Idle": (180, 180, 180),
        "No Person": (0, 0, 255),
    }

    def __init__(self):
        self.fps_smooth = 0.0

    def draw(self, frame: np.ndarray, action: str, fps: float, tracking: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        result = frame.copy()

        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        self.fps_smooth = config.FPS_SMOOTHING * self.fps_smooth + (1 - config.FPS_SMOOTHING) * fps

        cv2.putText(result, "VR ACTION RECOGNITION", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TITLE, 2)

        icon = self.ACTION_ICONS.get(action, "")
        action_color = self.ACTION_COLORS.get(action, config.COLOR_ACTION)
        cv2.putText(result, f"Action: {icon} {action}", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)

        cv2.putText(result, f"FPS: {int(self.fps_smooth)}", (w - 150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FPS, 2)

        status = "Tracking..." if tracking else "No Person Detected"
        status_color = config.COLOR_STATUS if tracking else (0, 0, 255)
        cv2.putText(result, f"Status: {status}", (15, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.putText(result, "ESC=quit  D=debug", (w - 220, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        return result
