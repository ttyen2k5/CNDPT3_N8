"""Phan 4: Game Controller - Ne chuong ngai vat bang Jump / Bend."""

import random
import cv2
import numpy as np
import config


class GameController:

    def __init__(self, panel_width=None, panel_height=None):
        self.panel_w = panel_width or config.GAME_PANEL_WIDTH
        self.panel_h = panel_height or config.CAMERA_HEIGHT
        self.ground_y = config.CHAR_GROUND_Y
        self.char_x = 150

        # Character physics
        self.char_y = self.ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.is_bending = False

        # Obstacles: list of {x, type}  type = "high" (need jump) or "low" (need bend)
        self.obstacles = []
        self.obstacle_speed = 8
        self.spawn_timer = 0
        self.spawn_interval = 60

        # Scoring
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.anim_frame = 0
        self.current_action = "Idle"

        # Hit flash effect
        self._hit_flash = 0

    def update(self, action):
        self.current_action = action
        self.anim_frame += 1

        if self.game_over:
            return

        # Jump
        if action == "Jump" and not self.is_jumping:
            self.is_jumping = True
            self.vel_y = config.CHAR_JUMP_VELOCITY

        # Bend
        self.is_bending = action == "Bend"

        # Jump physics
        if self.is_jumping:
            self.char_y += self.vel_y
            self.vel_y += config.CHAR_GRAVITY
            if self.char_y >= self.ground_y:
                self.char_y = self.ground_y
                self.is_jumping = False
                self.vel_y = 0

        # Spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            obs_type = random.choice(["high", "low"])
            self.obstacles.append({"x": self.panel_w + 20, "type": obs_type, "scored": False})
            self.spawn_interval = random.randint(50, 90)

        # Move obstacles & check collision
        alive_obs = []
        for obs in self.obstacles:
            obs["x"] -= self.obstacle_speed

            if obs["x"] < -40:
                continue

            # Score when obstacle passes the character
            if not obs["scored"] and obs["x"] + 30 < self.char_x:
                obs["scored"] = True
                self.score += 1

            # Collision detection
            if self._check_collision(obs):
                self.lives -= 1
                self._hit_flash = 12
                if self.lives <= 0:
                    self.game_over = True
                continue

            alive_obs.append(obs)

        self.obstacles = alive_obs

        if self._hit_flash > 0:
            self._hit_flash -= 1

        # Speed up gradually
        if self.score > 0 and self.score % 5 == 0:
            self.obstacle_speed = min(15, 8 + self.score // 5)

    def _check_collision(self, obs):
        ox = obs["x"]
        char_left = self.char_x - 15
        char_right = self.char_x + 15

        if ox + 30 < char_left or ox > char_right:
            return False

        if obs["type"] == "high":
            # High obstacle (bird/bar at head level) -> need to bend
            obs_top = self.ground_y - 60
            obs_bottom = self.ground_y - 20
            if self.is_bending:
                return False
            char_top = int(self.char_y) - 48
            return char_top < obs_bottom

        else:
            # Low obstacle (rock/spike on ground) -> need to jump
            obs_top = self.ground_y + 30
            obs_bottom = self.ground_y + 60
            char_bottom = int(self.char_y) + 60
            if self.is_jumping and self.char_y < self.ground_y - 20:
                return False
            return char_bottom > obs_top

    def draw(self):
        panel = np.full((self.panel_h, self.panel_w, 3), (30, 30, 50), dtype=np.uint8)

        # Sky gradient
        for row in range(0, self.ground_y + 60):
            t = row / (self.ground_y + 60)
            b = int(50 + 30 * t)
            g = int(30 + 20 * t)
            panel[row] = (b, g, 30)

        # Ground
        for row in range(self.ground_y + 60, self.panel_h):
            g = max(0, 60 - (row - self.ground_y - 60))
            panel[row] = (30 + g, 80 + g, 30)
        cv2.line(panel, (0, self.ground_y + 60), (self.panel_w, self.ground_y + 60), (50, 150, 50), 2)

        # Draw obstacles
        for obs in self.obstacles:
            self._draw_obstacle(panel, obs)

        # Draw character
        cx = self.char_x
        cy = int(self.char_y)
        if self._hit_flash > 0 and self._hit_flash % 2 == 0:
            pass
        else:
            self._draw_char(panel, cx, cy)

        # UI
        cv2.putText(panel, "DODGE GAME", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(panel, f"Score: {self.score}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Lives as hearts (left side, next to score)
        max_lives = 3
        for i in range(max_lives):
            hx = 140 + i * 25
            hy = 45
            alive = i < self.lives
            self._draw_heart(panel, hx, hy, size=9, alive=alive)

        # Action indicator
        action_color = (0, 255, 255) if self.current_action == "Idle" else \
                       (150, 255, 150) if self.current_action == "Jump" else \
                       (100, 255, 100) if self.current_action == "Bend" else (180, 180, 180)
        cv2.putText(panel, f"Action: {self.current_action}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color, 1)

        # Obstacle legend
        cv2.putText(panel, "^^^ = JUMP over", (10, self.panel_h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(panel, "--- = BEND under", (10, self.panel_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        if self.game_over:
            self._draw_game_over(panel)

        return panel

    def _draw_obstacle(self, panel, obs):
        ox = int(obs["x"])

        if obs["type"] == "low":
            # Ground spike / rock -> need to jump over
            pts = np.array([
                [ox, self.ground_y + 60],
                [ox + 15, self.ground_y + 25],
                [ox + 30, self.ground_y + 60],
            ], dtype=np.int32)
            cv2.fillPoly(panel, [pts], (80, 80, 220))
            cv2.polylines(panel, [pts], True, (100, 120, 255), 2)
        else:
            # High bar / bird -> need to bend under
            bar_y = self.ground_y - 40
            cv2.rectangle(panel, (ox, bar_y - 8), (ox + 40, bar_y + 8), (220, 80, 80), -1)
            cv2.rectangle(panel, (ox, bar_y - 8), (ox + 40, bar_y + 8), (255, 120, 120), 2)
            cv2.putText(panel, "---", (ox + 5, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)

    def _draw_char(self, panel, cx, cy):
        c = config.CHAR_COLOR
        hr = 16

        if self.is_bending:
            # Bent posture: head tilted forward, body horizontal
            cv2.circle(panel, (cx + 25, cy + 15), hr, c, 2)
            cv2.circle(panel, (cx + 21, cy + 12), 3, (255, 255, 255), -1)
            cv2.circle(panel, (cx + 29, cy + 12), 3, (255, 255, 255), -1)
            cv2.line(panel, (cx + 15, cy + 28), (cx, cy + 55), c, 3)
            cv2.line(panel, (cx, cy + 55), (cx - 12, cy + 60), c, 2)
            cv2.line(panel, (cx, cy + 55), (cx + 12, cy + 60), c, 2)
            cv2.line(panel, (cx + 10, cy + 35), (cx - 10, cy + 45), c, 2)
            cv2.line(panel, (cx + 10, cy + 35), (cx + 30, cy + 45), c, 2)
            return

        # Standing / jumping posture
        cv2.circle(panel, (cx, cy - 30), hr, c, 2)
        cv2.circle(panel, (cx - 4, cy - 33), 3, (255, 255, 255), -1)
        cv2.circle(panel, (cx + 4, cy - 33), 3, (255, 255, 255), -1)
        cv2.line(panel, (cx, cy - 14), (cx, cy + 40), c, 3)

        if self.is_jumping:
            cv2.line(panel, (cx, cy + 5), (cx - 25, cy - 20), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 25, cy - 20), c, 2)
        else:
            cv2.line(panel, (cx, cy + 5), (cx - 22, cy + 25), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 22, cy + 25), c, 2)

        cv2.line(panel, (cx, cy + 40), (cx - 15, cy + 60), c, 2)
        cv2.line(panel, (cx, cy + 40), (cx + 15, cy + 60), c, 2)

    def _draw_heart(self, panel, cx, cy, size=10, alive=True):
        s = size
        pts = np.array([
            [cx, cy + s],
            [cx - s, cy],
            [cx - s, cy - int(s * 0.5)],
            [cx - int(s * 0.6), cy - s],
            [cx - int(s * 0.2), cy - s],
            [cx, cy - int(s * 0.5)],
            [cx + int(s * 0.2), cy - s],
            [cx + int(s * 0.6), cy - s],
            [cx + s, cy - int(s * 0.5)],
            [cx + s, cy],
        ], dtype=np.int32)

        if alive:
            cv2.fillPoly(panel, [pts], (0, 0, 220))
            cv2.polylines(panel, [pts], True, (0, 0, 255), 1)
            highlight = np.array([
                [cx - int(s * 0.5), cy - int(s * 0.6)],
                [cx - int(s * 0.3), cy - int(s * 0.8)],
                [cx - int(s * 0.1), cy - int(s * 0.6)],
            ], dtype=np.int32)
            cv2.fillPoly(panel, [highlight], (80, 80, 255))
        else:
            cv2.polylines(panel, [pts], True, (80, 80, 80), 1)

    def _draw_game_over(self, panel):
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (self.panel_w, self.panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

        cx = self.panel_w // 2
        cy = self.panel_h // 2
        cv2.putText(panel, "GAME OVER", (cx - 100, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(panel, f"Score: {self.score}", (cx - 60, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Press R to restart", (cx - 90, cy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def restart(self):
        self.char_y = self.ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.is_bending = False
        self.obstacles.clear()
        self.spawn_timer = 0
        self.spawn_interval = 60
        self.obstacle_speed = 8
        self.score = 0
        self.lives = 3
        self.game_over = False
        self._hit_flash = 0
