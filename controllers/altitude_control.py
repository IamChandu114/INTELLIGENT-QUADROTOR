"""
Altitude PID Controller for Quadrotor Drone.
Intelligent Systems and Control (ISC) Project.

Regulates vehicle vertical altitude (Z) through dynamic thrust control.
Compensates for gravity, aerodynamic drag, and tilt angle loss of lift.
"""

import numpy as np
from drone.parameters import DroneParameters
from .pid_controller import PIDController


class AltitudePIDController:
    """Controls vertical altitude and calculates total collective thrust command T."""

    def __init__(
        self,
        params: DroneParameters | None = None,
        kp: float = 12.0,
        ki: float = 2.5,
        kd: float = 6.0
    ):
        self.params = params if params is not None else DroneParameters()
        
        # Max acceleration requested in Z: [-5 m/s^2, +10 m/s^2]
        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            output_min=-8.0,
            output_max=12.0,
            integrator_min=-3.0,
            integrator_max=3.0,
            name="Altitude_PID"
        )

    def reset(self) -> None:
        self.pid.reset()

    def update_gains(self, kp: float, ki: float, kd: float) -> None:
        self.pid.kp = float(kp)
        self.pid.ki = float(ki)
        self.pid.kd = float(kd)

    def compute_thrust(
        self,
        target_z: float,
        current_z: float,
        current_vz: float,
        pitch_theta: float,
        dt: float
    ) -> float:
        """
        Calculates total required thrust T [N].

        Args:
            target_z: Desired altitude setpoint [m]
            current_z: Actual vehicle altitude [m]
            current_vz: Vertical velocity [m/s]
            pitch_theta: Current vehicle tilt angle [rad]
            dt: Time step [s]

        Returns:
            Thrust T [N] within [thrust_min, thrust_max]
        """
        # Linear PID acceleration command
        a_z_des = self.pid.compute(
            target=target_z,
            current=current_z,
            dt=dt,
            feedforward=0.0,
            current_rate=current_vz
        )

        # Invert vertical dynamics:
        # z_ddot = (T / m) * cos(theta) - g  ==>  T = m * (g + a_z_des) / cos(theta)
        m = self.params.mass
        g = self.params.gravity

        # Prevent singularity when drone is severely tilted
        cos_theta = float(np.cos(pitch_theta))
        effective_cos = max(cos_theta, 0.3)

        nominal_thrust = (m * (g + a_z_des)) / effective_cos

        # Apply vehicle physical motor saturation limits
        thrust_clamped = float(np.clip(nominal_thrust, self.params.thrust_min, self.params.thrust_max))
        return thrust_clamped

