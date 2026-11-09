"""
Unified Quadrotor Simulation Engine.
Intelligent Systems and Control (ISC) Project.

Coordinates physical dynamics, intelligent cascaded controllers, telemetry logging,
and Pygame real-time rendering.
"""

from typing import List, Dict, Any, Optional
import os
import time
import numpy as np
import pandas as pd
import pygame

from drone.parameters import DroneParameters
from drone.dynamics import QuadrotorDynamics
from controllers.position_control import CascadedQuadrotorController, FlightMode
from .environment import SimulationEnvironment
from .drone_visualization import DroneVisualizer


class QuadrotorSimulator:
    """Master simulation harness for interactive and automated batch experiments."""

    def __init__(self, params: Optional[DroneParameters] = None):
        self.params = params if params is not None else DroneParameters()
        self.dynamics = QuadrotorDynamics(self.params)
        self.controller = CascadedQuadrotorController(self.params)

        # Telemetry logging buffer
        self.history: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.sim_time: float = 0.0

        # Visualization setup (lazy-initialized when interactive mode starts)
        self.env: Optional[SimulationEnvironment] = None
        self.visualizer: Optional[DroneVisualizer] = None
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

    def reset(self, initial_x: float = 0.0, initial_z: float = 0.0) -> None:
        """Reset drone state, controllers, and history buffer."""
        self.dynamics.reset(x=initial_x, z=initial_z)
        self.controller.reset()
        self.history.clear()
        self.sim_time = 0.0
        if self.visualizer is not None:
            self.visualizer.trajectory_points.clear()

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes one discrete simulation step of physics and intelligent feedback control.
        """
        dt = dt or self.params.dt

        # 1. Controller computes corrective thrust and torque from current state
        thrust_cmd, torque_cmd, desired_theta = self.controller.update(
            current_x=self.dynamics.x,
            current_z=self.dynamics.z,
            current_vx=self.dynamics.vx,
            current_vz=self.dynamics.vz,
            current_theta=self.dynamics.theta,
            current_omega=self.dynamics.omega,
            dt=dt
        )

        # 2. Physics engine integrates Runge-Kutta 4th order dynamics
        self.dynamics.step(thrust=thrust_cmd, torque=torque_cmd, dt=dt)
        self.sim_time += dt

        # 3. Telemetry capture
        wind_fx, _ = self.dynamics.wind.compute_force(self.sim_time)
        record = {
            "time": round(self.sim_time, 4),
            "x": self.dynamics.x,
            "z": self.dynamics.z,
            "vx": self.dynamics.vx,
            "vz": self.dynamics.vz,
            "theta": self.dynamics.theta,
            "theta_deg": self.dynamics.theta_deg,
            "omega": self.dynamics.omega,
            "omega_deg": float(np.degrees(self.dynamics.omega)),
            "target_x": self.controller.target_x,
            "target_z": self.controller.target_z,
            "desired_theta_deg": float(np.degrees(desired_theta)),
            "thrust": self.dynamics.thrust,
            "thrust_pct": (self.dynamics.thrust / self.params.thrust_max) * 100.0,
            "torque": self.dynamics.torque,
            "rotor_left": self.dynamics.rotor_thrust_left,
            "rotor_right": self.dynamics.rotor_thrust_right,
            "wind_fx": wind_fx,
            "wind_enabled": self.dynamics.wind.enabled,
            "mode": self.controller.mode.value
        }
        self.history.append(record)
        return record

    def toggle_wind(self, base_force: float = 2.0) -> bool:
        """Toggles wind disturbance with specified magnitude."""
        self.dynamics.wind.enabled = not self.dynamics.wind.enabled
        if self.dynamics.wind.enabled:
            self.dynamics.wind.base_force_x = base_force
        return self.dynamics.wind.enabled

    def get_history_df(self) -> pd.DataFrame:
        """Exports captured flight trajectory as a Pandas DataFrame."""
        return pd.DataFrame(self.history)

    def run_interactive(self, width: int = 1200, height: int = 750) -> None:
        """
        Launches the full real-time interactive Pygame simulation window.
        """
        # Set video driver to prevent errors on some Windows environments
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Real-Time Quadrotor Intelligent Control System (ISC Simulation)")

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.env = SimulationEnvironment(screen_width=width, screen_height=height)
        self.visualizer = DroneVisualizer(self.env)

        # Wire UI button callbacks
        actions = {
            "takeoff": lambda: self.controller.command_takeoff(target_altitude=5.0),
            "hover": lambda: self.controller.command_hover(self.dynamics.x, self.dynamics.z),
            "move_left": lambda: self.controller.command_move_relative(-2.0),
            "move_right": lambda: self.controller.command_move_relative(2.0),
            "land": lambda: self.controller.command_land(),
            "toggle_wind": lambda: self.toggle_wind(),
            "reset": lambda: self.reset(),
            "save_plots": lambda: self._trigger_plot_export(),
        }
        self.visualizer.setup_buttons(actions)

        self.is_running = True
        fps = 60
        substeps = int(round(1.0 / (fps * self.params.dt)))
        substeps = max(1, substeps)

        print("[Simulator] Pygame Interactive Simulation Running. Press ESC or close window to exit.")

        while self.is_running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key in (pygame.K_t, pygame.K_SPACE):
                        self.controller.command_takeoff(5.0)
                    elif event.key == pygame.K_l:
                        self.controller.command_land()
                    elif event.key == pygame.K_h:
                        self.controller.command_hover(self.dynamics.x, self.dynamics.z)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.controller.command_move_relative(-2.0)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.controller.command_move_relative(2.0)
                    elif event.key == pygame.K_w:
                        self.toggle_wind()
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_g:
                        self._trigger_plot_export()

                # Check UI button clicks
                self.visualizer.handle_event(event)

            # Advance physics (multiple sub-steps per frame to ensure continuous 100Hz fidelity at 60 FPS)
            for _ in range(substeps):
                telemetry = self.step(self.params.dt)

            # Smooth camera tracking
            self.env.camera_x += (self.dynamics.x - self.env.camera_x) * 0.05
            self.env.update_wind_particles(telemetry["wind_fx"], 1.0 / fps)
            self.visualizer.update_trajectory(self.dynamics.x, self.dynamics.z)

            # Rendering pipeline
            self.visualizer.draw_environment(self.screen)
            self.visualizer.draw_wind_particles(self.screen, telemetry["wind_fx"])
            self.visualizer.draw_trajectory(self.screen)
            self.visualizer.draw_target_marker(self.screen, self.controller.target_x, self.controller.target_z)
            self.visualizer.draw_quadrotor(
                surface=self.screen,
                x=self.dynamics.x,
                z=self.dynamics.z,
                theta=self.dynamics.theta,
                arm_length_m=self.params.arm_length,
                thrust_left=self.dynamics.rotor_thrust_left,
                thrust_right=self.dynamics.rotor_thrust_right,
                total_thrust=self.dynamics.thrust
            )
            self.visualizer.draw_telemetry_hud(self.screen, telemetry)

            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()
        print("[Simulator] Pygame Interactive Simulation terminated safely.")

    def _trigger_plot_export(self) -> None:
        """Trigger plot generation from current history."""
        from evaluation.plots import generate_all_plots
        from evaluation.metrics import calculate_flight_metrics
        df = self.get_history_df()
        if len(df) > 10:
            os.makedirs("results", exist_ok=True)
            generate_all_plots(df, output_dir="results")
            metrics = calculate_flight_metrics(df)
            print("[Simulator] Performance plots and metrics exported to 'results/' successfully.")
        else:
            print("[Simulator] Not enough simulation data recorded yet to export plots.")

