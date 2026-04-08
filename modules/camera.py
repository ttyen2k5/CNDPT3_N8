"""Phần 1: Module Camera - Mở webcam và đọc frame video."""

import cv2
import config


class Camera:
    """Quản lý webcam: mở, đọc frame, giải phóng."""

    def __init__(self, camera_id=None):
        self.camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        return self

    def read(self):
        """Đọc một frame từ camera. Trả về (success, frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
