"""Phan 4: Game Controller - Nhan vat demo phan hoi 6 hanh dong AI."""

import cv2
import numpy as np
import math
import config


class GameController:

    def __init__(self, panel_width=None, panel_height=None):
        self.panel_w = panel_width or config.GAME_PANEL_WIDTH
        self.panel_h = panel_height or config.CAMERA_HEIGHT
        self.ground_y = config.CHAR_GROUND_Y
        self.y = self.ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.run_offset = 0
        self.anim_frame = 0
        self.current_action = "Idle"
        self.menu_alpha = 0
        self.show_menu = False

    def update(self, action):
        self.current_action = action
        self.anim_frame += 1

        if action == "Jump" and not self.is_jumping:
            self.is_jumping = True
            self.vel_y = int(config.CHAR_JUMP_VELOCITY * 1.5)
        if action == "Running":
            self.run_offset = (self.run_offset + config.CHAR_RUN_SPEED) % (self.panel_w - 60)
        if action == "Walking":
            self.run_offset = (self.run_offset + config.CHAR_WALK_SPEED) % (self.panel_w - 60)

        if action == "Wave":
            self.menu_alpha = min(255, self.menu_alpha + 15)
            self.show_menu = True
        else:
            self.menu_alpha = max(0, self.menu_alpha - 10)
            if self.menu_alpha == 0:
                self.show_menu = False

        if self.is_jumping:
            self.y += self.vel_y
            self.vel_y += config.CHAR_GRAVITY
            if self.y >= self.ground_y:
                self.y = self.ground_y
                self.is_jumping = False
                self.vel_y = 0

    def draw(self):
        panel = np.full((self.panel_h, self.panel_w, 3), config.GAME_BG_COLOR, dtype=np.uint8)

        for row in range(self.ground_y + 60, self.panel_h):
            g = max(0, 60 - (row - self.ground_y - 60))
            panel[row] = (30 + g, 80 + g, 30)
        cv2.line(panel, (0, self.ground_y + 60), (self.panel_w, self.ground_y + 60), (50, 120, 50), 2)

        cx = 30 + self.run_offset
        cy = int(self.y)

        self._draw_char(panel, cx, cy)

        if self.show_menu and self.menu_alpha > 0:
            self._draw_menu(panel)

        cv2.putText(panel, "GAME DEMO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TITLE, 2)
        cv2.putText(panel, f"Action: {self.current_action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_ACTION, 1)
        return panel

    def _draw_char(self, panel, cx, cy):
        c = config.CHAR_COLOR
        act = self.current_action
        hr = 18

        if act == "Bend":
            cv2.circle(panel, (cx + 25, cy + 10), hr, c, 2)
            cv2.line(panel, (cx + 15, cy + 25), (cx, cy + 55), c, 3)
            cv2.line(panel, (cx, cy + 55), (cx - 10, cy + 60), c, 2)
            cv2.line(panel, (cx, cy + 55), (cx + 15, cy + 60), c, 2)
            return

        cv2.circle(panel, (cx, cy - 30), hr, c, 2)
        cv2.circle(panel, (cx - 5, cy - 34), 3, (255, 255, 255), -1)
        cv2.circle(panel, (cx + 5, cy - 34), 3, (255, 255, 255), -1)
        cv2.line(panel, (cx, cy - 12), (cx, cy + 40), c, 3)

        if act == "Jump":
            cv2.line(panel, (cx, cy + 5), (cx - 25, cy - 25), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 25, cy - 25), c, 2)
        elif act == "Raise Hand":
            cv2.line(panel, (cx, cy + 5), (cx - 20, cy - 30), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 20, cy - 30), c, 2)
        elif act == "Wave":
            wo = int(10 * math.sin(self.anim_frame * 0.5))
            cv2.line(panel, (cx, cy + 5), (cx - 25, cy + 20), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 25, cy - 20 + wo), c, 2)
        else:
            cv2.line(panel, (cx, cy + 5), (cx - 25, cy + 25), c, 2)
            cv2.line(panel, (cx, cy + 5), (cx + 25, cy + 25), c, 2)

        if act == "Running":
            lo = int(15 * math.sin(self.anim_frame * 0.4))
            cv2.line(panel, (cx, cy + 40), (cx - 15 + lo, cy + 60), c, 2)
            cv2.line(panel, (cx, cy + 40), (cx + 15 - lo, cy + 60), c, 2)
        elif act == "Walking":
            lo = int(8 * math.sin(self.anim_frame * 0.25))
            cv2.line(panel, (cx, cy + 40), (cx - 10 + lo, cy + 60), c, 2)
            cv2.line(panel, (cx, cy + 40), (cx + 10 - lo, cy + 60), c, 2)
        else:
            cv2.line(panel, (cx, cy + 40), (cx - 15, cy + 60), c, 2)
            cv2.line(panel, (cx, cy + 40), (cx + 15, cy + 60), c, 2)

    def _draw_menu(self, panel):
        a = self.menu_alpha / 255.0
        ov = panel.copy()
        mx, my, mw, mh = 80, 100, 240, 200
        cv2.rectangle(ov, (mx, my), (mx + mw, my + mh), (60, 60, 90), -1)
        cv2.rectangle(ov, (mx, my), (mx + mw, my + mh), (100, 200, 255), 2)
        for i, item in enumerate(["Resume Game", "Settings", "Inventory", "Exit"]):
            cv2.putText(ov, f"> {item}", (mx + 20, my + 45 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1)
        cv2.putText(ov, "MENU", (mx + 85, my + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        cv2.addWeighted(ov, a, panel, 1 - a, 0, panel)
