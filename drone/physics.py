"""
Physics, environmental disturbances, and ground interaction modeling.
Intelligent Systems and Control (ISC) Project.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class WindDisturbance:
    """Simulates realistic atmospheric wind disturbances and turbulent gusts."""
    enabled: bool = False
    base_force_x: float = 0.0          # Steady-state wind force in X direction [N]
    gust_amplitude: float = 2.5        # Peak gust amplitude [N]
    gust_frequency: float = 0.5        # Frequency of periodic wind gusts [Hz]
    turbulence_std: float = 0.3        # Standard deviation of high-frequency turbulence [N]
    vertical_draft: float = 0.0        # Vertical wind force (updraft/downdraft) [N]

    def compute_force(self, time: float) -> tuple[float, float]:
        """
        Calculate total aerodynamic disturbance force vector (Fx, Fz) at time t.
        
        Args:
            time: Current simulation time [s]
            
        Returns:
            Tuple of (Fx_wind, Fz_wind) [N]
        """
        if not self.enabled:
            return 0.0, 0.0

        # Periodic gust component + continuous base + stochastic turbulence
        periodic_gust = self.gust_amplitude * np.sin(2.0 * np.pi * self.gust_frequency * time)
        stochastic_noise = np.random.normal(0.0, self.turbulence_std)
        fx = self.base_force_x + periodic_gust + stochastic_noise

        # Minor vertical atmospheric fluctuation
        fz = self.vertical_draft + 0.15 * stochastic_noise
        return float(fx), float(fz)


class GroundPhysics:
    """Handles drone physical boundary conditions with the ground plane (z = 0)."""

    @staticmethod
    def resolve_ground_contact(
        z: float,
        vz: float,
        vx: float,
        theta: float,
        omega: float,
        thrust: float,
        hover_thrust: float,
        dt: float
    ) -> tuple[float, float, float, float, float]:
        """
        Applies normal reaction, inelastic restitution, and surface friction when on ground.

        Returns:
            Tuple of updated (z, vz, vx, theta, omega)
        """
        if z <= 0.0:
            z = 0.0
            if vz < 0.0:
                vz = 0.0  # Stop downward velocity upon impact

            # If motors are disarmed or thrust is well below hover thrust, drone rests stably on ground
            if thrust < 0.6 * hover_thrust:
                vz = 0.0
                # Surface friction dampens horizontal slide
                vx *= max(0.0, 1.0 - 15.0 * dt)
                # Ground contact levels out pitch attitude
                theta *= max(0.0, 1.0 - 20.0 * dt)
                omega = 0.0
                if abs(vx) < 1e-3:
                    vx = 0.0
                if abs(theta) < 1e-4:
                    theta = 0.0

        return z, vz, vx, theta, omega

