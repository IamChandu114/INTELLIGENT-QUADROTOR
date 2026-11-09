"""
Interactive Pygame Visualization for Quadrotor Intelligent Control System.
Intelligent Systems and Control (ISC) Project.

Features:
- High-fidelity quadrotor rendering with 4 spinning rotors, chassis, and LED lights
- Thrust vectors proportional to individual motor forces
- Target waypoint crosshair with pulsing indicator
- Smooth historical flight trajectory ribbon
- Atmospheric wind particle flow and vector arrow
- Ground runway, distance grid, and altitude ruler markings
- Dark aerospace HUD telemetry panel
- Clickable on-screen control buttons and keyboard shortcuts
"""

from typing import List, Tuple, Dict, Any, Callable
import math
import numpy as np
import pygame
from .environment import SimulationEnvironment
from controllers.position_control import FlightMode


class UIButton:
    """Interactive clickable on-screen button."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Callable[[], None],
        bg_color: Tuple[int, int, int] = (30, 42, 56),
        hover_color: Tuple[int, int, int] = (45, 65, 88),
        text_color: Tuple[int, int, int] = (220, 230, 242)
    ):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = self.hover_color if self.is_hovered else self.bg_color
        # Rounded button background
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (80, 110, 140), self.rect, width=1, border_radius=6)

        txt_surf = font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class DroneVisualizer:
    """Renders the drone, physics environment, telemetry HUD, and interactive UI."""

    def __init__(self, env: SimulationEnvironment):
        self.env = env
        self.propeller_angle: float = 0.0
        self.trajectory_points: List[Tuple[float, float]] = []
        self.max_trajectory_length: int = 400

        # Colors (Dark Theme Aerospace Palette)
        self.COLOR_BG = (15, 20, 28)
        self.COLOR_GRID = (28, 38, 52)
        self.COLOR_GROUND = (35, 45, 58)
        self.COLOR_RUNWAY = (50, 60, 75)
        self.COLOR_TEXT = (210, 225, 240)
        self.COLOR_TEXT_DIM = (130, 150, 175)
        self.COLOR_CHASSIS = (180, 190, 205)
        self.COLOR_ARMS = (70, 85, 105)
        self.COLOR_ROTOR = (52, 152, 219)
        self.COLOR_ROTOR_BLUR = (52, 152, 219, 140)
        self.COLOR_THRUST = (231, 76, 60)
        self.COLOR_TARGET = (46, 204, 113)
        self.COLOR_TRAIL = (41, 128, 185)
        self.COLOR_HUD_BG = (22, 30, 42, 220)
        self.COLOR_PANEL_BORDER = (45, 60, 80)

        # Pygame Fonts
        self.font_main = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15)
        self.font_bold = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12)
        self.font_btn = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13, bold=True)

        self.buttons: List[UIButton] = []

    def setup_buttons(self, actions: Dict[str, Callable[[], None]]) -> None:
        """Create clickable UI buttons at top-right banner."""
        btn_width = 95
        btn_height = 30
        spacing = 8
        start_x = 20
        y = 15

        button_defs = [
            ("Takeoff", actions.get("takeoff"), (26, 92, 56)),
            ("Hover", actions.get("hover"), (35, 75, 115)),
            ("<- Left", actions.get("move_left"), (45, 60, 85)),
            ("Right ->", actions.get("move_right"), (45, 60, 85)),
            ("Land", actions.get("land"), (120, 80, 30)),
            ("Wind Tgl", actions.get("toggle_wind"), (90, 50, 110)),
            ("Reset", actions.get("reset"), (110, 40, 40)),
            ("Save Plots", actions.get("save_plots"), (30, 80, 100)),
        ]

        self.buttons.clear()
        for i, (label, cb, color) in enumerate(button_defs):
            if cb is not None:
                rect = pygame.Rect(start_x + i * (btn_width + spacing), y, btn_width, btn_height)
                self.buttons.append(UIButton(rect, label, cb, bg_color=color))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process mouse clicks on UI elements."""
        handled = False
        for btn in self.buttons:
            if btn.handle_event(event):
                handled = True
        return handled

    def update_trajectory(self, x: float, z: float) -> None:
        """Append drone coordinates to historical flight path trail."""
        self.trajectory_points.append((x, z))
        if len(self.trajectory_points) > self.max_trajectory_length:
            self.trajectory_points.pop(0)

    def draw_environment(self, surface: pygame.Surface) -> None:
        """Draw sky background, ground plane, altitude ruler, and helipad."""
        surface.fill(self.COLOR_BG)

        # Draw Altitude Grid Lines every 1.0 meter
        for alt_m in range(1, 11):
            _, sy = self.env.world_to_screen(0.0, float(alt_m))
            pygame.draw.line(surface, self.COLOR_GRID, (0, sy), (self.env.screen_width, sy), 1)
            # Altitude label
            lbl = self.font_small.render(f"{alt_m}.0 m", True, self.COLOR_TEXT_DIM)
            surface.blit(lbl, (self.env.screen_width - 55, sy - 8))

        # Ground Plane
        ground_rect = pygame.Rect(
            0,
            self.env.ground_y,
            self.env.screen_width,
            self.env.screen_height - self.env.ground_y
        )
        pygame.draw.rect(surface, self.COLOR_GROUND, ground_rect)
        pygame.draw.line(
            surface,
            (70, 95, 125),
            (0, self.env.ground_y),
            (self.env.screen_width, self.env.ground_y),
            3
        )

        # Runway & Helipad at Origin (x = 0)
        h_center_x, h_center_y = self.env.world_to_screen(0.0, 0.0)
        pad_width = self.env.meters_to_pixels(1.5)
        pad_rect = pygame.Rect(h_center_x - pad_width // 2, self.env.ground_y - 2, pad_width, 10)
        pygame.draw.rect(surface, (85, 105, 130), pad_rect)
        pygame.draw.circle(surface, (230, 200, 50), (h_center_x, self.env.ground_y + 18), 16, 2)
        h_text = self.font_bold.render("H", True, (230, 200, 50))
        surface.blit(h_text, (h_center_x - 6, self.env.ground_y + 9))

        # Distance Grid Tick Marks every 1.0 m on Ground
        for x_m in range(-15, 16):
            if x_m == 0:
                continue
            sx, _ = self.env.world_to_screen(float(x_m), 0.0)
            if 0 <= sx <= self.env.screen_width:
                pygame.draw.line(surface, (55, 75, 100), (sx, self.env.ground_y), (sx, self.env.ground_y + 10), 2)
                t = self.font_small.render(f"{x_m}m", True, self.COLOR_TEXT_DIM)
                surface.blit(t, (sx - 10, self.env.ground_y + 12))

    def draw_wind_particles(self, surface: pygame.Surface, wind_fx: float) -> None:
        """Render floating wind streak lines."""
        if abs(wind_fx) < 0.05:
            return

        for p in self.env.wind_particles:
            alpha = min(255, int(p["alpha"] * (abs(wind_fx) / 2.0)))
            streak_len = p["length"] * (1.0 + abs(wind_fx) * 0.5)
            dx = streak_len if wind_fx > 0 else -streak_len

            start_pt = (int(p["x"]), int(p["y"]))
            end_pt = (int(p["x"] + dx), int(p["y"]))

            # Draw particle line
            color = (130, 190, 240)
            pygame.draw.line(surface, color, start_pt, end_pt, 1)

    def draw_trajectory(self, surface: pygame.Surface) -> None:
        """Draw historical flight path ribbon."""
        if len(self.trajectory_points) < 2:
            return

        screen_pts = [self.env.world_to_screen(x, z) for x, z in self.trajectory_points]
        for i in range(1, len(screen_pts)):
            ratio = i / len(screen_pts)
            color = (
                int(30 + 100 * ratio),
                int(100 + 100 * ratio),
                int(180 + 75 * ratio)
            )
            pygame.draw.line(surface, color, screen_pts[i - 1], screen_pts[i], 2)

    def draw_target_marker(self, surface: pygame.Surface, target_x: float, target_z: float) -> None:
        """Render animated pulsing crosshair at commanded target waypoint."""
        tx, ty = self.env.world_to_screen(target_x, target_z)

        # Pulsing circle
        t = pygame.time.get_ticks() / 1000.0
        pulse_r = int(14 + 4 * math.sin(6.0 * t))

        # Reticle
        pygame.draw.circle(surface, self.COLOR_TARGET, (tx, ty), pulse_r, 1)
        pygame.draw.circle(surface, self.COLOR_TARGET, (tx, ty), 3)
        pygame.draw.line(surface, self.COLOR_TARGET, (tx - 18, ty), (tx + 18, ty), 1)
        pygame.draw.line(surface, self.COLOR_TARGET, (tx, ty - 18), (tx, ty + 18), 1)

        # Target label
        lbl = self.font_small.render(f"WP: ({target_x:.1f}m, {target_z:.1f}m)", True, self.COLOR_TARGET)
        surface.blit(lbl, (tx + 15, ty - 22))

    def draw_quadrotor(
        self,
        surface: pygame.Surface,
        x: float,
        z: float,
        theta: float,
        arm_length_m: float,
        thrust_left: float,
        thrust_right: float,
        total_thrust: float
    ) -> None:
        """
        Draws the quadrotor vehicle body, 4 rotor discs, tilt angle orientation,
        and motor thrust force vectors.
        """
        cx, cy = self.env.world_to_screen(x, z)
        arm_px = self.env.meters_to_pixels(arm_length_m)

        # Propeller rotation update (spin speed depends on thrust)
        self.propeller_angle += (15.0 + total_thrust * 2.5)

        # Rotate drone body based on pitch angle theta (radians)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Arm endpoints relative to center of mass
        # Drone coordinate frame: Left arm (-arm_px), Right arm (+arm_px)
        lx = cx - int(arm_px * cos_t)
        ly = cy + int(arm_px * sin_t)
        rx = cx + int(arm_px * cos_t)
        ry = cy - int(arm_px * sin_t)

        # 1. Main Structural Carbon Fiber Arm Rod
        pygame.draw.line(surface, self.COLOR_ARMS, (lx, ly), (rx, ry), 6)
        pygame.draw.line(surface, (120, 140, 165), (lx, ly), (rx, ry), 2)

        # 2. Central Avionics Fuselage Pod
        pod_w, pod_h = 24, 14
        # Create a rotated surface for the center pod
        pod_surf = pygame.Surface((pod_w * 2, pod_h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(pod_surf, (35, 45, 60), (pod_w // 2, pod_h // 2, pod_w, pod_h))
        pygame.draw.ellipse(pod_surf, (80, 190, 240), (pod_w // 2, pod_h // 2, pod_w, pod_h), 2)
        # Orientation heading nose LED (green)
        pygame.draw.circle(pod_surf, (46, 204, 113), (pod_w // 2 + pod_w - 4, pod_h), 3)

        rotated_pod = pygame.transform.rotate(pod_surf, math.degrees(-theta))
        surface.blit(rotated_pod, rotated_pod.get_rect(center=(cx, cy)))

        # 3. Four Rotor Discs (2 on Left Arm, 2 on Right Arm in planar view)
        motor_radius = 8
        rotor_blade_radius = 22

        rotor_positions = [
            (lx, ly, thrust_left, "L"),
            (rx, ry, thrust_right, "R")
        ]

        # Normal vector pointing upward perpendicular to the drone arm (in screen space)
        # Arm vector is (cos_t, -sin_t). Perpendicular upward is (sin_t, cos_t) in standard,
        # but in screen coordinates: up is -Y: (-sin_t, -cos_t)
        norm_x = -sin_t
        norm_y = -cos_t

        for r_x, r_y, r_thrust, side in rotor_positions:
            # Motor Mount Pylon
            pygame.draw.circle(surface, (50, 60, 75), (r_x, r_y), motor_radius)
            pygame.draw.circle(surface, (190, 200, 215), (r_x, r_y), motor_radius, 2)

            # Rotor Shaft
            shaft_len = 6
            top_x = r_x + int(shaft_len * norm_x)
            top_y = r_y + int(shaft_len * norm_y)
            pygame.draw.line(surface, (140, 150, 165), (r_x, r_y), (top_x, top_y), 3)

            # Spinning Propeller Blades
            spin_rad = math.radians(self.propeller_angle if side == "R" else -self.propeller_angle)
            p_dx = int(rotor_blade_radius * math.cos(spin_rad) * cos_t)
            p_dy = int(-rotor_blade_radius * math.cos(spin_rad) * sin_t)

            pygame.draw.line(
                surface,
                self.COLOR_ROTOR,
                (top_x - p_dx, top_y - p_dy),
                (top_x + p_dx, top_y + p_dy),
                3
            )
            # Propeller safety guard ring
            pygame.draw.ellipse(
                surface,
                (70, 130, 180),
                (top_x - rotor_blade_radius, top_y - 4, rotor_blade_radius * 2, 8),
                1
            )

            # Navigation LEDs: Red for Left, Green for Right
            led_color = (231, 76, 60) if side == "L" else (46, 204, 113)
            pygame.draw.circle(surface, led_color, (r_x, r_y), 3)

            # 4. Motor Thrust Force Vector Arrows
            if r_thrust > 0.5:
                thrust_px = int(r_thrust * 4.5)  # Arrow length proportional to thrust
                arr_x = top_x + int(thrust_px * norm_x)
                arr_y = top_y + int(thrust_px * norm_y)
                pygame.draw.line(surface, self.COLOR_THRUST, (top_x, top_y), (arr_x, arr_y), 3)
                # Arrowhead
                pygame.draw.circle(surface, (255, 140, 0), (arr_x, arr_y), 4)

    def draw_telemetry_hud(
        self,
        surface: pygame.Surface,
        telemetry: Dict[str, Any]
    ) -> None:
        """Render high-tech aerospace telemetry HUD cards at top-left and bottom."""
        # Top-Left Telemetry Box
        box_w, box_h = 360, 240
        box_x, box_y = 20, 60

        hud_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, self.COLOR_HUD_BG, (0, 0, box_w, box_h), border_radius=8)
        pygame.draw.rect(hud_surf, self.COLOR_PANEL_BORDER, (0, 0, box_w, box_h), width=1, border_radius=8)
        surface.blit(hud_surf, (box_x, box_y))

        # Title and Mode Badge
        title = self.font_title.render("QUADROTOR TELEMETRY", True, (100, 200, 255))
        surface.blit(title, (box_x + 14, box_y + 12))

        mode_str = telemetry.get("mode", "UNKNOWN")
        mode_color = (46, 204, 113) if mode_str == "HOVER" else (
            (52, 152, 219) if mode_str == "TAKEOFF" else (
                (241, 196, 15) if mode_str == "TRACKING" else (
                    (230, 126, 34) if mode_str == "LANDING" else (231, 76, 60)
                )
            )
        )
        badge = self.font_bold.render(f"[{mode_str}]", True, mode_color)
        surface.blit(badge, (box_x + box_w - badge.get_width() - 14, box_y + 14))

        # Telemetry Data Lines
        lines = [
            ("Altitude (Z):", f"{telemetry.get('z', 0.0):.2f} m  (Target: {telemetry.get('target_z', 0.0):.1f} m)"),
            ("Position (X):", f"{telemetry.get('x', 0.0):.2f} m  (Target: {telemetry.get('target_x', 0.0):.1f} m)"),
            ("Vertical Vel (Vz):", f"{telemetry.get('vz', 0.0):+.2f} m/s"),
            ("Horizontal Vel (Vx):", f"{telemetry.get('vx', 0.0):+.2f} m/s"),
            ("Pitch Angle (Theta):", f"{telemetry.get('theta_deg', 0.0):+.2f}°  (Des: {telemetry.get('desired_theta_deg', 0.0):+.2f}°)"),
            ("Angular Rate (Omega):", f"{telemetry.get('omega_deg', 0.0):+.2f}°/s"),
            ("Total Thrust (T):", f"{telemetry.get('thrust', 0.0):.2f} N  ({telemetry.get('thrust_pct', 0.0):.1f}% Max)"),
            ("Pitch Torque (Tau):", f"{telemetry.get('torque', 0.0):+.3f} N*m"),
        ]

        line_y = box_y + 44
        for label, val in lines:
            lbl_surf = self.font_small.render(label, True, self.COLOR_TEXT_DIM)
            val_surf = self.font_bold.render(val, True, self.COLOR_TEXT)
            surface.blit(lbl_surf, (box_x + 14, line_y))
            surface.blit(val_surf, (box_x + 160, line_y))
            line_y += 21

        # Wind Disturbance Indicator Card
        wind_active = telemetry.get("wind_enabled", False)
        wind_fx = telemetry.get("wind_fx", 0.0)
        w_box_w, w_box_h = 240, 60
        w_box_x = box_x
        w_box_y = box_y + box_h + 10

        w_hud = pygame.Surface((w_box_w, w_box_h), pygame.SRCALPHA)
        pygame.draw.rect(w_hud, self.COLOR_HUD_BG, (0, 0, w_box_w, w_box_h), border_radius=8)
        pygame.draw.rect(w_hud, self.COLOR_PANEL_BORDER, (0, 0, w_box_w, w_box_h), width=1, border_radius=8)
        surface.blit(w_hud, (w_box_x, w_box_y))

        w_status = "ACTIVE" if wind_active else "OFF"
        w_color = (231, 76, 60) if wind_active else (130, 150, 175)
        w_title = self.font_bold.render(f"WIND DISTURBANCE: {w_status}", True, w_color)
        surface.blit(w_title, (w_box_x + 12, w_box_y + 10))

        w_val = self.font_small.render(f"Force Fx: {wind_fx:+.2f} N  | Key: 'W'", True, self.COLOR_TEXT)
        surface.blit(w_val, (w_box_x + 12, w_box_y + 32))

        # Draw UI Buttons
        for btn in self.buttons:
            btn.draw(surface, self.font_btn)

        # Hotkeys Helper Banner at Bottom
        keys_text = (
            "HOTKEYS:  [T/Space]: Takeoff  |  [L]: Land  |  [H]: Hover  |  "
            "[A/Left]: Move -X  |  [D/Right]: Move +X  |  [W]: Toggle Wind  |  [R]: Reset  |  [G]: Export Graphs"
        )
        keys_surf = self.font_small.render(keys_text, True, (160, 185, 215))
        surface.blit(keys_surf, (20, self.env.screen_height - 30))

