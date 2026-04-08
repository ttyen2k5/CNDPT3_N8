"""Phan 3: Action Recognition - Nhan dien 6 hanh dong tu keypoints."""

from collections import deque
import config


class ActionRecognizer:
    """
    6 hanh dong:
      Raise Hand | Wave | Running | Walking | Jump | Bend
    """

    def __init__(self):
        self.wrist_history = deque(maxlen=config.WAVE_HISTORY_SIZE)
        self.ankle_history = deque(maxlen=config.RUN_HISTORY_SIZE)
        self.hip_history = deque(maxlen=config.JUMP_HISTORY_SIZE)
        self.current_action = "Idle"
        self._hold_counter = 0
        self._prev_raw = "Idle"
        self._wave_sticky = 0

    def recognize(self, keypoints):
        if not keypoints:
            self.current_action = "No Person"
            return self.current_action

        self._collect_history(keypoints)

        checks = [
            ("Wave",       self._is_wave(keypoints)),
            ("Raise Hand", self._is_raise_hand(keypoints)),
            ("Running",    self._is_running()),
            ("Walking",    self._is_walking()),
            ("Jump",       self._is_jump()),
            ("Bend",       self._is_bend(keypoints)),
        ]

        raw = "Idle"
        for name, detected in checks:
            if detected:
                raw = name
                break

        if raw == "Raise Hand":
            self.hip_history.clear()

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
        la = kp.get("left_ankle")
        ra = kp.get("right_ankle")
        if la and ra:
            self.ankle_history.append((la[1], ra[1]))
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if lh and rh:
            self.hip_history.append((lh[1] + rh[1]) / 2)

    # 1. RAISE HAND
    def _is_raise_hand(self, kp):
        th = config.RAISE_HAND_THRESHOLD
        return (self._above(kp, "left_wrist", "left_shoulder", th) or
                self._above(kp, "right_wrist", "right_shoulder", th))

    # 2. WAVE
    def _is_wave(self, kp):
        hand_up = self._is_raise_hand(kp)
        if not hand_up:
            self.wrist_history.clear()
            self._wave_sticky = 0
            return False

        lw = kp.get("left_wrist", (0, 1, 0, 0))
        rw = kp.get("right_wrist", (0, 1, 0, 0))
        self.wrist_history.append(lw[0] if lw[1] < rw[1] else rw[0])

        if self._wave_sticky > 0:
            self._wave_sticky -= 1
            if self._check_wave_pattern():
                self._wave_sticky = config.WAVE_STICKY_FRAMES
            return True

        if len(self.wrist_history) < 8:
            return False

        if self._check_wave_pattern():
            self._wave_sticky = config.WAVE_STICKY_FRAMES
            return True
        return False

    def _check_wave_pattern(self):
        if len(self.wrist_history) < 8:
            return False
        h = list(self.wrist_history)
        if max(h) - min(h) < config.WAVE_MIN_RANGE:
            return False
        vel = [h[i] - h[i - 1] for i in range(1, len(h))]
        dz = config.WAVE_DEAD_ZONE
        sc = 0
        for i in range(1, len(vel)):
            if (vel[i - 1] > dz and vel[i] < -dz) or \
               (vel[i - 1] < -dz and vel[i] > dz):
                sc += 1
        return sc >= config.WAVE_MIN_SIGN_CHANGES

    # 3. RUNNING
    def _is_running(self):
        if len(self.ankle_history) < 12:
            return False
        h = list(self.ankle_history)
        l_ys = [p[0] for p in h]
        r_ys = [p[1] for p in h]
        l_range = max(l_ys) - min(l_ys)
        r_range = max(r_ys) - min(r_ys)
        if l_range < config.RUN_MIN_ANKLE_RANGE or r_range < config.RUN_MIN_ANKLE_RANGE:
            return False
        l_speed = sum(abs(l_ys[i] - l_ys[i-1]) for i in range(1, len(l_ys))) / len(l_ys)
        r_speed = sum(abs(r_ys[i] - r_ys[i-1]) for i in range(1, len(r_ys))) / len(r_ys)
        if l_speed < config.RUN_MIN_SPEED or r_speed < config.RUN_MIN_SPEED:
            return False
        diff = [l_ys[i] - r_ys[i] for i in range(len(h))]
        st = config.RUN_SIGN_THRESHOLD
        sc = sum(1 for i in range(1, len(diff))
                 if (diff[i-1] > st and diff[i] < -st) or
                    (diff[i-1] < -st and diff[i] > st))
        return sc >= config.RUN_MIN_ALTERNATIONS

    # 4. WALKING
    def _is_walking(self):
        if len(self.ankle_history) < 12:
            return False
        h = list(self.ankle_history)
        l_ys = [p[0] for p in h]
        r_ys = [p[1] for p in h]
        l_range = max(l_ys) - min(l_ys)
        r_range = max(r_ys) - min(r_ys)
        if l_range < config.WALK_MIN_ANKLE_RANGE or r_range < config.WALK_MIN_ANKLE_RANGE:
            return False
        if l_range > config.WALK_MAX_ANKLE_RANGE and r_range > config.WALK_MAX_ANKLE_RANGE:
            return False
        diff = [l_ys[i] - r_ys[i] for i in range(len(h))]
        st = config.WALK_SIGN_THRESHOLD
        sc = sum(1 for i in range(1, len(diff))
                 if (diff[i-1] > st and diff[i] < -st) or
                    (diff[i-1] < -st and diff[i] > st))
        return sc >= config.WALK_MIN_ALTERNATIONS

    # 5. JUMP
    def _is_jump(self):
        if len(self.hip_history) < 10:
            return False
        h = list(self.hip_history)
        baseline = sum(h[:len(h)//2]) / (len(h)//2)
        return baseline - h[-1] > config.JUMP_MIN_RISE

    # 6. BEND
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

    @staticmethod
    def _above(kp, joint_a, joint_b, threshold):
        a = kp.get(joint_a)
        b = kp.get(joint_b)
        if not a or not b:
            return False
        return a[1] < b[1] - threshold

    def get_debug_info(self, keypoints):
        if not keypoints:
            return {}
        info = {}

        for side in ("left", "right"):
            w = keypoints.get(f"{side}_wrist")
            s = keypoints.get(f"{side}_shoulder")
            if w and s:
                info[f"{side[0].upper()}_hand"] = f"up:{s[1]-w[1]:+.2f} ext:{abs(w[0]-s[0]):.2f}"

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
                info["bend"] = f"{(hip_y - nose[1]) / st:.2f}"

        if len(self.ankle_history) >= 8:
            h = list(self.ankle_history)[-15:]
            ly = [p[0] for p in h]
            ry = [p[1] for p in h]
            info["ankle"] = f"L:{max(ly)-min(ly):.3f} R:{max(ry)-min(ry):.3f}"

        if len(self.hip_history) >= 10:
            h = list(self.hip_history)
            base = sum(h[:len(h)//2]) / (len(h)//2)
            info["jump"] = f"{base - h[-1]:+.3f}"

        if self._wave_sticky > 0:
            info["wave_s"] = f"{self._wave_sticky}"

        return info
