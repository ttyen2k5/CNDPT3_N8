"""
VR Action Recognition System - 10 Actions
===========================================
Camera -> Pose Estimation -> Action Recognition -> Game Control -> Display

Actions:
  Raise Hand -> jump       |  Wave    -> menu
  T-Pose     -> shield     |  Punch   -> attack
  Kick       -> kick       |  Pick Up -> collect item
  Running    -> run fast   |  Walking -> walk slow
  Jump       -> high jump  |  Bend    -> bend down

Press ESC to quit.  Press D to toggle debug overlay.
"""

import sys
import time
import cv2
import numpy as np

import config
from modules import Camera, PoseEstimator, ActionRecognizer, GameController, UIOverlay


def draw_debug(frame, debug_info, action):
    """Draw debug panel on bottom-left of camera frame."""
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

    print("[OK] System ready. Press ESC to quit, D to toggle debug.")
    prev_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("[WARNING] Cannot read frame.")
                break

            frame = cv2.flip(frame, 1)

            # Pose Estimation
            keypoints, annotated_frame = pose_estimator.process(frame)
            tracking = len(keypoints) > 0

            # Action Recognition
            action = action_recognizer.recognize(keypoints)

            # Debug overlay
            if show_debug and tracking:
                debug_info = action_recognizer.get_debug_info(keypoints)
                draw_debug(annotated_frame, debug_info, action)

            # Game update
            game.update(action)
            game_panel = game.draw()

            # UI overlay
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            display_frame = ui.draw(annotated_frame, action, fps, tracking)

            # Combine camera + game panel
            cam_h, cam_w = display_frame.shape[:2]
            game_h, game_w = game_panel.shape[:2]
            if game_h != cam_h:
                game_panel = cv2.resize(game_panel, (game_w, cam_h))

            combined = np.hstack([display_frame, game_panel])
            cv2.imshow(config.WINDOW_NAME, combined)

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
