"""
Neon Runner - Pose Estimation Game
====================================
Camera -> Pose Estimation -> Action Recognition -> Game -> Display

Actions:
  Jump  -> Nhan vat nhay qua chuong ngai vat
  Bend  -> Nhan vat cui ne drone

Press ESC to quit.  Press D to toggle debug overlay.
"""

import sys
import time
import cv2
import numpy as np

import config
from modules import Camera, PoseEstimator, ActionRecognizer, GameController, UIOverlay


def draw_debug(frame, debug_info, action):
    """Draw debug panel on bottom-left of frame."""
    h, w = frame.shape[:2]
    x0, y0 = 10, h - 20 * (len(debug_info) + 1) - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0 - 5), (320, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"[DEBUG] action={action}", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for i, (k, v) in enumerate(debug_info.items()):
        ty = y0 + 20 * (i + 1)
        cv2.putText(frame, f"  {k}: {v}", (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)


def main():
    camera = Camera()
    pose_estimator = PoseEstimator()
    action_recognizer = ActionRecognizer()
    game = GameController(panel_width=config.CAMERA_WIDTH,
                          panel_height=config.CAMERA_HEIGHT)
    ui = UIOverlay()
    show_debug = config.SHOW_DEBUG

    try:
        camera.open()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("[OK] Neon Runner ready. Press ESC to quit, D to toggle debug.")
    prev_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("[WARNING] Cannot read frame.")
                break

            frame = cv2.flip(frame, 1)

            keypoints, annotated_frame = pose_estimator.process(frame)
            tracking = len(keypoints) > 0
            action = action_recognizer.recognize(keypoints)

            game.update(action)
            game_panel = game.draw()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cam_display = ui.draw(annotated_frame, action, fps, tracking)

            pip_w = config.PIP_WIDTH
            pip_h = config.PIP_HEIGHT
            margin = config.PIP_MARGIN
            pip = cv2.resize(cam_display, (pip_w, pip_h))

            gp_h, gp_w = game_panel.shape[:2]
            px = gp_w - pip_w - margin
            py = 65

            cv2.rectangle(game_panel,
                          (px - 2, py - 2),
                          (px + pip_w + 1, py + pip_h + 1),
                          config.COLOR_RUNNER, 2)
            game_panel[py:py + pip_h, px:px + pip_w] = pip

            if show_debug and tracking:
                debug_info = action_recognizer.get_debug_info(keypoints)
                draw_debug(game_panel, debug_info, action)

            cv2.imshow(config.WINDOW_NAME, game_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[OK] Exited.")
                break
            elif key == ord('d') or key == ord('D'):
                show_debug = not show_debug

    except KeyboardInterrupt:
        print("\n[OK] Stopped by Ctrl+C.")
    finally:
        camera.release()
        pose_estimator.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
