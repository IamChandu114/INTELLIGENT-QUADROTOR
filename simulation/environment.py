"""
Simulation environment, coordinate projections, and atmospheric wind particles.
Intelligent Systems and Control (ISC) Project.
"""

from typing import Tuple, List
import numpy as np


class SimulationEnvironment:
    """Manages spatial transformation between physical SI units (meters) and Pygame pixels."""

    def __init__(
        self,
        screen_width: int = 1200,
        screen_height: int = 750,
        pixels_per_meter: float = 65.0,
        ground_margin_bottom: int = 100
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = pixels_per_meter
        self.ground_y = screen_height - ground_margin_bottom
        self.center_x = screen_width // 2

        # Camera panning tracking
        self.camera_x = 0.0

        # Dynamic wind visual particles
        self.wind_particles: List[dict] = []
        self._init_wind_particles(num_particles=60)

    def _init_wind_particles(self, num_particles: int) -> None:
        """Initialize random floating particles for atmospheric visualization."""
        for _ in range(num_particles):
            self.wind_particles.append({
                "x": np.random.uniform(0, self.screen_width),
                "y": np.random.uniform(50, self.ground_y - 20),
                "speed": np.random.uniform(0.8, 2.2),
                "length": np.random.uniform(10, 25),
                "alpha": np.random.randint(40, 140)
            })

    def update_wind_particles(self, wind_force_x: float, dt: float) -> None:
        """Update particle positions based on current wind speed and direction."""
        if abs(wind_force_x) < 0.05:
            return

        direction = 1.0 if wind_force_x > 0 else -1.0
        magnitude = min(abs(wind_force_x) * 120.0, 500.0)

        for p in self.wind_particles:
            p["x"] += direction * magnitude * p["speed"] * dt
            # Wrap around screen edges
            if p["x"] > self.screen_width:
                p["x"] = 0
                p["y"] = np.random.uniform(50, self.ground_y - 20)
            elif p["x"] < 0:
                p["x"] = self.screen_width
                p["y"] = np.random.uniform(50, self.ground_y - 20)

    def world_to_screen(self, x: float, z: float) -> Tuple[int, int]:
        """
        Converts physical coordinates (X, Z) in meters to screen pixel coordinates (u, v).
        
        Args:
            x: Horizontal position [m] (positive right)
            z: Vertical altitude [m] (positive up, ground at z = 0)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        sx = int(self.center_x + (x - self.camera_x) * self.scale)
        sy = int(self.ground_y - z * self.scale)
        return sx, sy

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Converts screen pixel coordinates back to physical world coordinates (m)."""
        x = (screen_x - self.center_x) / self.scale + self.camera_x
        z = max(0.0, (self.ground_y - screen_y) / self.scale)
        return x, z

    def meters_to_pixels(self, length_meters: float) -> int:
        """Scales a physical length in meters to screen pixels."""
        return int(length_meters * self.scale)

