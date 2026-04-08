"""Phan 3: Action Recognition - Nhan dien 2 hanh dong: Jump va Bend."""

from collections import deque
import config


class ActionRecognizer:
    """
    2 hanh dong dung cho Dino Runner:
      Jump  - nguoi choi nhay len
      Bend  - nguoi choi cui nguoi xuong
    """

    def __init__(self):
        self.hip_history = deque(maxlen=config.JUMP_HISTORY_SIZE)
        self.current_action = "Idle"
        self._hold_counter = 0
        self._prev_raw = "Idle"

    def recognize(self, keypoints):
        if not keypoints:
            self.current_action = "No Person"
            return self.current_action

        self._collect_history(keypoints)

        if self._is_jump():
            raw = "Jump"
        elif self._is_bend(keypoints):
            raw = "Bend"
        else:
            raw = "Idle"

        if raw != self._prev_raw:
            self._hold_counter += 1
            if self._hold_counter >= config.ACTION_HOLD_FRAMES:
                self.current_action = raw
                self._prev_raw = raw
                self._hold_counter = 0
        else:
            self._hold_counter = 0
            self.current_action = raw

        return self.current_action

    def _collect_history(self, kp):
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if lh and rh:
            self.hip_history.append((lh[1] + rh[1]) / 2)

    def _is_jump(self):
        if len(self.hip_history) < 12:
            return False
        h = list(self.hip_history)
        half = len(h) // 2
        baseline = sum(h[:half]) / half
        recent_avg = sum(h[-3:]) / 3
        return baseline - recent_avg > config.JUMP_MIN_RISE

    def _is_bend(self, kp):
        nose = kp.get("nose")
        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if not all([nose, ls, rs, lh, rh]):
            return False
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        sh_to_hip = hip_y - shoulder_y
        if sh_to_hip < 0.03:
            return True
        return (hip_y - nose[1]) / sh_to_hip < config.BEND_HEIGHT_RATIO

    def get_debug_info(self, keypoints):
        if not keypoints:
            return {}
        info = {}

        nose = keypoints.get("nose")
        ls = keypoints.get("left_shoulder")
        rs = keypoints.get("right_shoulder")
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")
        if all([nose, ls, rs, lh, rh]):
            sh_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            st = hip_y - sh_y
            if st > 0.03:
                ratio = (hip_y - nose[1]) / st
                info["bend"] = f"{ratio:.2f} (thr {config.BEND_HEIGHT_RATIO})"

        if len(self.hip_history) >= 12:
            h = list(self.hip_history)
            half = len(h) // 2
            base = sum(h[:half]) / half
            recent = sum(h[-3:]) / 3
            info["jump"] = f"{base - recent:+.3f} (thr {config.JUMP_MIN_RISE})"

        return info
