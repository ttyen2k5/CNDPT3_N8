"""Phan 5: UI Overlay - Hien thi thong tin hanh dong, FPS, trang thai."""

import cv2
import numpy as np
import config


class UIOverlay:
    """Ve overlay thong tin len frame camera."""

    def __init__(self):
        self.fps_smooth = 0.0

    def draw(self, frame: np.ndarray, action: str, fps: float, tracking: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        result = frame.copy()

        self.fps_smooth = config.FPS_SMOOTHING * self.fps_smooth + (1 - config.FPS_SMOOTHING) * fps

        # Draw bottom bar
        cv2.rectangle(result, (0, h - 40), (w, h), (0, 0, 0), -1)

        cv2.putText(result, f"FPS: {int(self.fps_smooth)}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FPS, 2)

        status = "Tracking..." if tracking else "No Person Detected"
        status_color = config.COLOR_STATUS if tracking else (0, 0, 255)
        cv2.putText(result, f"Status: {status}", (150, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.putText(result, "ESC=quit  D=debug  R=restart", (w - 350, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return result
