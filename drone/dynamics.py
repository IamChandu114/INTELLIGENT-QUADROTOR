"""
Quadrotor planar 2D flight dynamics and equations of motion.
Intelligent Systems and Control (ISC) Project.

State Vector:
    x: Horizontal position [m]
    z: Vertical altitude [m] (z >= 0)
    vx: Horizontal velocity [m/s]
    vz: Vertical velocity [m/s]
    theta: Pitch attitude angle [rad] (positive tilts drone right / counter-clockwise)
    omega: Angular velocity [rad/s]
"""

from typing import Tuple, List
import numpy as np
from .parameters import DroneParameters
from .physics import WindDisturbance, GroundPhysics


class QuadrotorDynamics:
    """Simulates 2D quadrotor equations of motion using Runge-Kutta 4th Order numerical integration."""

    def __init__(self, params: DroneParameters | None = None):
        self.params = params if params is not None else DroneParameters()
        self.ground = GroundPhysics()
        
        # State vector: [x, z, vx, vz, theta, omega]
        self.state = np.zeros(6, dtype=float)
        
        # Actuation state
        self.thrust: float = 0.0     # Total collective thrust [N]
        self.torque: float = 0.0     # Pitch control moment [N*m]
        
        # Rotor forces for visual feedback
        self.rotor_thrust_left: float = 0.0
        self.rotor_thrust_right: float = 0.0

        # Environmental wind
        self.wind = WindDisturbance()
        self.time: float = 0.0

    def reset(self, x: float = 0.0, z: float = 0.0, theta: float = 0.0) -> None:
        """Reset drone state to specified initial conditions."""
        self.state = np.array([x, z, 0.0, 0.0, theta, 0.0], dtype=float)
        self.thrust = 0.0
        self.torque = 0.0
        self.rotor_thrust_left = 0.0
        self.rotor_thrust_right = 0.0
        self.time = 0.0

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def z(self) -> float:
        return float(self.state[1])

    @property
    def vx(self) -> float:
        return float(self.state[2])

    @property
    def vz(self) -> float:
        return float(self.state[3])

    @property
    def theta(self) -> float:
        return float(self.state[4])

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.state[4]))

    @property
    def omega(self) -> float:
        return float(self.state[5])

    def _state_derivative(
        self,
        state: np.ndarray,
        thrust: float,
        torque: float,
        f_wind_x: float,
        f_wind_z: float
    ) -> np.ndarray:
        """
        Computes the time derivative of the continuous state vector: dx/dt = f(x, u, d).

        Equations of Motion:
            x_dot = vx
            z_dot = vz
            vx_dot = - (T / m) * sin(theta) - (c_dx / m) * vx + (Fx_wind / m)
            vz_dot =   (T / m) * cos(theta) - g - (c_dz / m) * vz + (Fz_wind / m)
            theta_dot = omega
            omega_dot = (tau - c_rot * omega) / I_yy
        """
        _, z, vx, vz, theta, omega = state
        m = self.params.mass
        g = self.params.gravity
        I = self.params.I_yy
        c_rot = self.params.rotational_damping
        c_dx = self.params.linear_drag_x
        c_dz = self.params.linear_drag_z

        # Kinematic rates
        x_dot = vx
        z_dot = vz

        # Accelerations
        vx_dot = -(thrust / m) * np.sin(theta) - (c_dx / m) * vx + (f_wind_x / m)
        vz_dot = (thrust / m) * np.cos(theta) - g - (c_dz / m) * vz + (f_wind_z / m)

        # Angular acceleration
        theta_dot = omega
        omega_dot = (torque - c_rot * omega) / I

        # If on ground resting and thrust is zero, prevent downward acceleration
        if z <= 0.0 and vz_dot < 0.0:
            vz_dot = 0.0

        return np.array([x_dot, z_dot, vx_dot, vz_dot, theta_dot, omega_dot], dtype=float)

    def step(self, thrust: float, torque: float, dt: float | None = None) -> np.ndarray:
        """
        Advances the physical simulation by one timestep using 4th-order Runge-Kutta (RK4).

        Args:
            thrust: Commanded total thrust [N]
            torque: Commanded pitch moment [N*m]
            dt: Timestep duration [s] (defaults to params.dt)

        Returns:
            Updated state vector np.ndarray
        """
        dt = dt or self.params.dt

        # Actuator Saturation Clamping
        self.thrust = float(np.clip(thrust, self.params.thrust_min, self.params.thrust_max))
        self.torque = float(np.clip(torque, -self.params.torque_max, self.params.torque_max))

        # Individual rotor thrust calculation:
        # T = f_left + f_right
        # tau = (f_right - f_left) * l  ==>  tau / l = f_right - f_left
        l = self.params.arm_length
        diff = self.torque / (2.0 * l)
        half_thrust = self.thrust / 2.0
        self.rotor_thrust_left = max(0.0, half_thrust - diff)
        self.rotor_thrust_right = max(0.0, half_thrust + diff)

        # Environmental Disturbances
        f_wind_x, f_wind_z = self.wind.compute_force(self.time)

        # RK4 Integration
        k1 = self._state_derivative(self.state, self.thrust, self.torque, f_wind_x, f_wind_z)
        k2 = self._state_derivative(self.state + 0.5 * dt * k1, self.thrust, self.torque, f_wind_x, f_wind_z)
        k3 = self._state_derivative(self.state + 0.5 * dt * k2, self.thrust, self.torque, f_wind_x, f_wind_z)
        k4 = self._state_derivative(self.state + dt * k3, self.thrust, self.torque, f_wind_x, f_wind_z)

        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.time += dt

        # Resolve Ground Physics Boundaries
        z, vz, vx, theta, omega = self.ground.resolve_ground_contact(
            z=self.state[1],
            vz=self.state[3],
            vx=self.state[2],
            theta=self.state[4],
            omega=self.state[5],
            thrust=self.thrust,
            hover_thrust=self.params.hover_thrust,
            dt=dt
        )
        self.state[1] = z
        self.state[3] = vz
        self.state[2] = vx
        self.state[4] = theta
        self.state[5] = omega

        # Numerical safety: constrain attitude to [-pi, pi]
        self.state[4] = (self.state[4] + np.pi) % (2.0 * np.pi) - np.pi

        return self.state.copy()
