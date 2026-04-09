"""
Action Recognition - Dodge Game
================================
Camera -> Pose Estimation -> ML Model (Jump/Bend) -> Game Control -> Display

Jump  = nhay de ne chuong ngai vat tren mat dat
Bend  = cui de ne chuong ngai vat tren cao
Idle  = dung yen (model khong du tu tin)

Press ESC to quit.  Press D to toggle debug.  Press R to restart game.
"""

import sys
import time
import cv2
import numpy as np

import config
from modules import Camera, PoseEstimator, ActionRecognizer, GameController, UIOverlay


def draw_debug(frame, debug_info, action):
    h, w = frame.shape[:2]
    x0, y0 = 10, h - 20 * (len(debug_info) + 1) - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0 - 5), (320, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"[DEBUG] raw={action}", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for i, (k, v) in enumerate(debug_info.items()):
        ty = y0 + 20 * (i + 1)
        cv2.putText(frame, f"  {k}: {v}", (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)


def main():
    camera = Camera()
    pose_estimator = PoseEstimator()
    action_recognizer = ActionRecognizer()
    game = GameController(panel_height=config.CAMERA_HEIGHT)
    ui = UIOverlay()
    show_debug = config.SHOW_DEBUG

    try:
        camera.open()
        print("[OK] Camera opened successfully.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("[OK] System ready. Press ESC to quit, D to toggle debug, R to restart game.")
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

            if show_debug and tracking:
                debug_info = action_recognizer.get_debug_info(keypoints)
                draw_debug(annotated_frame, debug_info, action)

            game.update(action)
            main_screen = game.draw()

            # Resize camera frame for PiP (Picture-in-Picture)
            pip_w = 320
            pip_h = int(pip_w * (annotated_frame.shape[0] / annotated_frame.shape[1]))
            pip_frame = cv2.resize(annotated_frame, (pip_w, pip_h))
            
            # Put PiP in top-right corner
            margin = 20
            main_screen[margin:margin+pip_h, main_screen.shape[1]-pip_w-margin:main_screen.shape[1]-margin] = pip_frame
            cv2.rectangle(main_screen, 
                          (main_screen.shape[1]-pip_w-margin, margin), 
                          (main_screen.shape[1]-margin, margin+pip_h), 
                          (255, 255, 255), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            display_frame = ui.draw(main_screen, action, fps, tracking)

            if show_debug and tracking:
                debug_info = action_recognizer.get_debug_info(keypoints)
                draw_debug(display_frame, debug_info, action)

            cv2.imshow(config.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[OK] Exited.")
                break
            elif key == ord('d') or key == ord('D'):
                show_debug = not show_debug
            elif key == ord('r') or key == ord('R'):
                game.restart()

    except KeyboardInterrupt:
        print("\n[OK] Stopped by Ctrl+C.")
    finally:
        camera.release()
        pose_estimator.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
