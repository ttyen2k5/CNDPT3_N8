"""Phan 4: Game Controller - Neon Runner dieu khien bang Pose Estimation."""

import cv2
import numpy as np
import random
import math
import config


class GameController:

    def __init__(self, panel_width=None, panel_height=None):
        self.panel_w = panel_width or config.CAMERA_WIDTH
        self.panel_h = panel_height or config.CAMERA_HEIGHT
        self.hi_score = 0
        self._init_stars()
        self.reset()

    def _init_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                "x": random.randint(0, self.panel_w),
                "y": random.randint(15, config.GROUND_Y - 40),
                "size": random.choice([1, 1, 1, 2]),
                "bright": random.randint(60, 200),
            })

    def reset(self):
        self.dino_y = float(config.GROUND_Y)
        self.dino_vel_y = 0.0
        self.is_jumping = False
        self.is_ducking = False
        self.anim_frame = 0

        self.state = "playing"
        self.score = 0
        self.speed = config.INITIAL_SPEED
        self.game_over_timer = 0

        self.obstacles = []
        self.spawn_timer = 50

        self.ground_offset = 0.0
        self.score_flash = 0
        self.trail = []

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------

    def update(self, action):
        self.anim_frame += 1

        self.trail = [p for p in self.trail if p["life"] > 0]
        for p in self.trail:
            p["x"] -= self.speed * 0.4
            p["life"] -= 1

        if self.state == "game_over":
            self.game_over_timer += 1
            if self.is_jumping:
                self.dino_y += self.dino_vel_y
                self.dino_vel_y += config.GRAVITY
                if self.dino_y >= config.GROUND_Y:
                    self.dino_y = float(config.GROUND_Y)
                    self.is_jumping = False
            if (self.game_over_timer > config.GAME_OVER_RESTART_FRAMES
                    and action == "Jump"):
                self.reset()
            return

        if len(self.trail) < 60:
            ty = int(self.dino_y) - (10 if self.is_ducking else 20)
            self.trail.append({
                "x": float(config.RUNNER_X - 5),
                "y": ty + random.randint(-5, 5),
                "life": 14,
            })

        if action == "Jump" and not self.is_jumping:
            self.is_jumping = True
            self.dino_vel_y = config.JUMP_VELOCITY

        self.is_ducking = (action == "Bend") and not self.is_jumping

        if action == "Bend" and self.is_jumping:
            self.dino_vel_y += config.GRAVITY

        if self.is_jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y += config.GRAVITY
            if self.dino_y >= config.GROUND_Y:
                self.dino_y = float(config.GROUND_Y)
                self.is_jumping = False
                self.dino_vel_y = 0.0

        for obs in self.obstacles:
            obs["x"] -= self.speed
        self.obstacles = [o for o in self.obstacles if o["x"] > -80]

        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_obstacle()
            gap_px = random.randint(config.OBSTACLE_MIN_GAP,
                                    config.OBSTACLE_MAX_GAP)
            self.spawn_timer = max(15, int(gap_px / self.speed))

        if self._check_collision():
            self.state = "game_over"
            self.hi_score = max(self.hi_score, self.score)
            return

        self.score += 1
        if self.score > 0 and self.score % 500 == 0:
            self.score_flash = 30
        if self.score_flash > 0:
            self.score_flash -= 1
        self.speed = min(config.MAX_SPEED,
                         self.speed + config.SPEED_INCREMENT)

        for star in self.stars:
            star["x"] -= self.speed * 0.15
            if star["x"] < -5:
                star["x"] = self.panel_w + random.randint(5, 50)
                star["y"] = random.randint(15, config.GROUND_Y - 40)

        self.ground_offset = (self.ground_offset + self.speed) % 60

    # ------------------------------------------------------------------
    #  Spawning & collision
    # ------------------------------------------------------------------

    def _spawn_obstacle(self):
        gy = config.GROUND_Y
        r = random.random()

        if r < 0.25 and self.speed > config.DRONE_MIN_SPEED:
            self.obstacles.append({
                "type": "drone", "x": float(self.panel_w + 20),
                "y": gy - 60, "w": 40, "h": 25,
                "anim": random.randint(0, 100),
            })
        elif r < 0.50:
            self.obstacles.append({
                "type": "barrier_tall", "x": float(self.panel_w + 20),
                "y": gy - 50, "w": 18, "h": 50,
            })
        elif r < 0.75:
            self.obstacles.append({
                "type": "barrier_small", "x": float(self.panel_w + 20),
                "y": gy - 35, "w": 15, "h": 35,
            })
        else:
            self.obstacles.append({
                "type": "barrier_group", "x": float(self.panel_w + 20),
                "y": gy - 45, "w": 38, "h": 45,
            })

    def _get_hitbox(self):
        dx = config.RUNNER_X
        dy = int(self.dino_y)
        if self.is_ducking:
            return (dx, dy - 22, dx + 24, dy + 4)
        return (dx, dy - 48, dx + 24, dy)

    def _check_collision(self):
        d = self._get_hitbox()
        for obs in self.obstacles:
            ox, oy = int(obs["x"]), int(obs["y"])
            o = (ox, oy, ox + obs["w"], oy + obs["h"])
            if d[0] < o[2] and d[2] > o[0] and d[1] < o[3] and d[3] > o[1]:
                return True
        return False

    # ------------------------------------------------------------------
    #  Glow helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _glow_rect(panel, p1, p2, color, glow=3):
        dim = tuple(max(0, c // 4) for c in color)
        cv2.rectangle(panel,
                      (p1[0] - glow, p1[1] - glow),
                      (p2[0] + glow, p2[1] + glow), dim, -1)
        cv2.rectangle(panel, p1, p2, color, -1)

    @staticmethod
    def _glow_circle(panel, center, radius, color, glow=3):
        dim = tuple(max(0, c // 4) for c in color)
        cv2.circle(panel, center, radius + glow, dim, -1)
        cv2.circle(panel, center, radius, color, -1)

    @staticmethod
    def _glow_line(panel, p1, p2, color, thickness=2, glow=2):
        dim = tuple(max(0, c // 4) for c in color)
        cv2.line(panel, p1, p2, dim, thickness + glow * 2)
        cv2.line(panel, p1, p2, color, thickness)

    # ------------------------------------------------------------------
    #  Main draw
    # ------------------------------------------------------------------

    def draw(self):
        panel = np.full((self.panel_h, self.panel_w, 3),
                        config.GAME_BG_COLOR, dtype=np.uint8)

        self._draw_stars(panel)
        self._draw_ground(panel)
        self._draw_trail(panel)

        for obs in self.obstacles:
            self._draw_obstacle(panel, obs)

        self._draw_character(panel)
        self._draw_hud(panel)

        if self.state == "game_over":
            self._draw_game_over(panel)

        return panel

    # ------------------------------------------------------------------
    #  Environment
    # ------------------------------------------------------------------

    def _draw_stars(self, panel):
        for star in self.stars:
            sx = int(star["x"]) % self.panel_w
            b = star["bright"]
            cv2.circle(panel, (sx, star["y"]), star["size"],
                       (b, b, min(255, b + 30)), -1)

    def _draw_ground(self, panel):
        gy = config.GROUND_Y
        gc = config.COLOR_GROUND

        dim = tuple(max(0, c // 4) for c in gc)
        cv2.line(panel, (0, gy + 2), (self.panel_w, gy + 2), dim, 5)
        cv2.line(panel, (0, gy + 2), (self.panel_w, gy + 2), gc, 1)

        for i in range(1, 8):
            y = gy + 2 + i * 22
            if y >= self.panel_h:
                break
            intensity = max(0, 70 - i * 9)
            cv2.line(panel, (0, y), (self.panel_w, y),
                     (0, intensity, 0), 1)

        offset = int(self.ground_offset)
        for x_base in range(-60, self.panel_w + 60, 60):
            x = x_base - offset
            if 0 <= x < self.panel_w:
                cv2.line(panel, (x, gy + 2),
                         (x, min(self.panel_h, gy + 160)),
                         (0, 30, 12), 1)

    def _draw_trail(self, panel):
        c_base = config.COLOR_RUNNER
        for p in self.trail:
            if p["life"] > 0:
                a = p["life"] / 14.0
                color = tuple(int(c * a * 0.35) for c in c_base)
                r = max(1, int(3 * a))
                cv2.circle(panel, (int(p["x"]), p["y"]), r, color, -1)

    # ------------------------------------------------------------------
    #  Character
    # ------------------------------------------------------------------

    def _draw_character(self, panel):
        c = config.COLOR_RUNNER
        dx = config.RUNNER_X
        dy = int(self.dino_y)

        if self.is_ducking and not self.is_jumping:
            self._draw_char_duck(panel, dx, dy, c)
        else:
            self._draw_char_stand(panel, dx, dy, c)

    def _draw_char_stand(self, panel, dx, dy, c):
        bright = tuple(min(255, v + 60) for v in c)

        self._glow_circle(panel, (dx + 12, dy - 42), 12, c, glow=4)

        if self.state == "game_over":
            cv2.line(panel, (dx + 8, dy - 47), (dx + 14, dy - 41),
                     config.COLOR_GAME_OVER, 2)
            cv2.line(panel, (dx + 8, dy - 41), (dx + 14, dy - 47),
                     config.COLOR_GAME_OVER, 2)
        else:
            cv2.rectangle(panel, (dx + 15, dy - 47), (dx + 22, dy - 41),
                          config.COLOR_VISOR, -1)
            cv2.rectangle(panel, (dx + 18, dy - 46), (dx + 21, dy - 42),
                          (255, 255, 255), -1)

        self._glow_rect(panel, (dx + 2, dy - 30), (dx + 22, dy - 8),
                        c, glow=3)
        cv2.rectangle(panel, (dx + 10, dy - 28), (dx + 14, dy - 10),
                      bright, -1)

        sw = int(5 * math.sin(self.anim_frame * 0.3))
        self._glow_line(panel, (dx + 2, dy - 25),
                        (dx - 10, dy - 18 + sw), c, 2, 1)
        self._glow_line(panel, (dx + 22, dy - 25),
                        (dx + 34, dy - 18 - sw), c, 2, 1)

        if self.is_jumping or self.state == "game_over":
            self._glow_line(panel, (dx + 6, dy - 8),
                            (dx, dy + 8), c, 2, 1)
            self._glow_line(panel, (dx + 18, dy - 8),
                            (dx + 24, dy + 8), c, 2, 1)
        else:
            lo = int(8 * math.sin(self.anim_frame * 0.4))
            self._glow_line(panel, (dx + 6, dy - 8),
                            (dx + lo, dy + 8), c, 2, 1)
            self._glow_line(panel, (dx + 18, dy - 8),
                            (dx + 24 - lo, dy + 8), c, 2, 1)

    def _draw_char_duck(self, panel, dx, dy, c):
        """Crouching pose — head on top, body compressed, knees bent."""
        bright = tuple(min(255, v + 60) for v in c)

        # Head (on top, lowered)
        self._glow_circle(panel, (dx + 12, dy - 20), 10, c, glow=3)
        cv2.rectangle(panel, (dx + 14, dy - 24), (dx + 20, dy - 19),
                      config.COLOR_VISOR, -1)
        cv2.rectangle(panel, (dx + 16, dy - 23), (dx + 19, dy - 20),
                      (255, 255, 255), -1)

        # Compact body
        self._glow_rect(panel, (dx, dy - 10), (dx + 24, dy + 2), c, glow=3)
        cv2.rectangle(panel, (dx + 10, dy - 8), (dx + 14, dy), bright, -1)

        # Arms bent forward
        self._glow_line(panel, (dx + 3, dy - 7),
                        (dx - 8, dy - 2), c, 2, 1)
        self._glow_line(panel, (dx + 21, dy - 7),
                        (dx + 32, dy - 2), c, 2, 1)

        # Bent knees
        lo = 4 if (self.anim_frame // 6) % 2 == 0 else 0
        self._glow_line(panel, (dx + 4, dy + 2),
                        (dx - 2 + lo, dy + 10), c, 2, 1)
        self._glow_line(panel, (dx + 20, dy + 2),
                        (dx + 26 - lo, dy + 10), c, 2, 1)

    # ------------------------------------------------------------------
    #  Obstacles
    # ------------------------------------------------------------------

    def _draw_obstacle(self, panel, obs):
        x, y = int(obs["x"]), int(obs["y"])
        t = obs["type"]

        if t == "drone":
            c = config.COLOR_DRONE
            self._glow_rect(panel, (x, y + 6), (x + 30, y + 18), c, glow=3)
            wing_up = (self.anim_frame + obs.get("anim", 0)) // 8 % 2 == 0
            if wing_up:
                self._glow_line(panel, (x + 8, y + 6),
                                (x + 15, y - 6), c, 2, 2)
                self._glow_line(panel, (x + 22, y + 6),
                                (x + 15, y - 6), c, 2, 2)
            else:
                self._glow_line(panel, (x + 8, y + 18),
                                (x + 15, y + 28), c, 2, 2)
                self._glow_line(panel, (x + 22, y + 18),
                                (x + 15, y + 28), c, 2, 2)
            cv2.circle(panel, (x + 26, y + 12), 3, (255, 255, 255), -1)

        elif t == "barrier_tall":
            c = config.COLOR_BARRIER
            self._glow_rect(panel, (x + 3, y), (x + 15, y + 50), c, glow=3)
            self._glow_rect(panel, (x, y - 3), (x + 18, y + 3), c, glow=2)
            bright = tuple(min(255, v + 80) for v in c)
            for i in range(0, 44, 10):
                cv2.line(panel, (x + 5, y + 5 + i), (x + 13, y + 5 + i),
                         bright, 1)

        elif t == "barrier_small":
            c = config.COLOR_BARRIER
            self._glow_rect(panel, (x + 2, y), (x + 13, y + 35), c, glow=3)
            self._glow_rect(panel, (x - 1, y - 2), (x + 16, y + 3), c,
                            glow=2)

        elif t == "barrier_group":
            c = config.COLOR_BARRIER
            self._glow_rect(panel, (x + 2, y + 5), (x + 12, y + 45), c, 2)
            self._glow_rect(panel, (x + 16, y), (x + 26, y + 45), c, 2)
            self._glow_rect(panel, (x + 30, y + 8), (x + 38, y + 45), c, 2)
            cv2.line(panel, (x + 12, y + 20), (x + 16, y + 18), c, 2)
            cv2.line(panel, (x + 26, y + 22), (x + 30, y + 20), c, 2)

    # ------------------------------------------------------------------
    #  HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, panel):
        sc = config.COLOR_SCORE
        if self.score_flash > 0 and self.score_flash % 6 < 3:
            sc = config.GAME_BG_COLOR

        px = self.panel_w - 160
        cv2.rectangle(panel, (px - 10, 8), (self.panel_w - 8, 52),
                      (30, 20, 40), -1)
        cv2.rectangle(panel, (px - 10, 8), (self.panel_w - 8, 52),
                      (70, 50, 100), 1)

        cv2.putText(panel, "SCORE", (px, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 140, 180), 1)
        cv2.putText(panel, f"{self.score:05d}", (px, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)

        if self.hi_score > 0:
            hx = px - 150
            cv2.putText(panel, "BEST", (hx, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 140, 180), 1)
            cv2.putText(panel, f"{self.hi_score:05d}", (hx, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 110, 140), 2)

        spd_pct = ((self.speed - config.INITIAL_SPEED)
                    / max(1, config.MAX_SPEED - config.INITIAL_SPEED))
        bw, bx, by = 80, 15, 15
        cv2.rectangle(panel, (bx, by), (bx + bw, by + 8), (40, 30, 55), -1)
        fw = int(bw * spd_pct)
        if fw > 0:
            r = int(255 * spd_pct)
            g = int(255 * (1 - spd_pct))
            cv2.rectangle(panel, (bx, by), (bx + fw, by + 8), (0, g, r), -1)
        cv2.putText(panel, "SPEED", (bx, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 110, 140), 1)

    # ------------------------------------------------------------------
    #  Game Over
    # ------------------------------------------------------------------

    def _draw_game_over(self, panel):
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (self.panel_w, self.panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)

        c = config.COLOR_GAME_OVER
        text = "GAME OVER"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        tx = (self.panel_w - ts[0]) // 2
        ty = self.panel_h // 2 - 30

        dim = tuple(v // 3 for v in c)
        cv2.putText(panel, text, (tx - 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, dim, 5)
        cv2.putText(panel, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)

        stxt = f"SCORE  {self.score:05d}"
        ss = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(panel, stxt, ((self.panel_w - ss[0]) // 2, ty + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.game_over_timer > config.GAME_OVER_RESTART_FRAMES:
            pulse = int(abs(math.sin(self.anim_frame * 0.08)) * 80) + 175
            hint = "JUMP TO RESTART"
            hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(panel, hint,
                        ((self.panel_w - hs[0]) // 2, ty + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, pulse, pulse), 1)
