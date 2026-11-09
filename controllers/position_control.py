"""
Cascaded Horizontal Position and Attitude Flight Control.
Intelligent Systems and Control (ISC) Project.

Architecture:
  [Outer Loop] Position Controller: Computes desired tilt angle theta_des from position error.
  [Inner Loop] Attitude Controller: Regulates pitch angle to theta_des by commanding torque tau.
"""

from enum import Enum
import numpy as np
from drone.parameters import DroneParameters
from .pid_controller import PIDController
from .altitude_control import AltitudePIDController


class FlightMode(str, Enum):
    """Operational flight state machine modes."""
    DISARMED = "DISARMED"
    TAKEOFF = "TAKEOFF"
    HOVER = "HOVER"
    TRACKING = "TRACKING"
    LANDING = "LANDING"
    LANDED = "LANDED"
    EMERGENCY = "EMERGENCY"


class PositionAttitudeController:
    """Cascaded outer-loop position and inner-loop attitude controller."""

    def __init__(
        self,
        params: DroneParameters | None = None,
        pos_kp: float = 1.8,
        pos_ki: float = 0.1,
        pos_kd: float = 2.2,
        att_kp: float = 8.0,
        att_ki: float = 0.5,
        att_kd: float = 1.2
    ):
        self.params = params if params is not None else DroneParameters()

        # Outer Loop: Position PID -> outputs desired horizontal acceleration [m/s^2]
        self.pos_pid = PIDController(
            kp=pos_kp,
            ki=pos_ki,
            kd=pos_kd,
            output_min=-3.5,
            output_max=3.5,
            integrator_min=-1.0,
            integrator_max=1.0,
            name="Position_Outer_PID"
        )

        # Inner Loop: Attitude PID -> outputs corrective rotational torque [N*m]
        self.att_pid = PIDController(
            kp=att_kp,
            ki=att_ki,
            kd=att_kd,
            output_min=-self.params.torque_max,
            output_max=self.params.torque_max,
            integrator_min=-0.5,
            integrator_max=0.5,
            name="Attitude_Inner_PID"
        )

        self.desired_theta: float = 0.0

    def reset(self) -> None:
        self.pos_pid.reset()
        self.att_pid.reset()
        self.desired_theta = 0.0

    def update_gains(
        self,
        pos_kp: float,
        pos_ki: float,
        pos_kd: float,
        att_kp: float,
        att_ki: float,
        att_kd: float
    ) -> None:
        self.pos_pid.kp = float(pos_kp)
        self.pos_pid.ki = float(pos_ki)
        self.pos_pid.kd = float(pos_kd)
        self.att_pid.kp = float(att_kp)
        self.att_pid.ki = float(att_ki)
        self.att_pid.kd = float(att_kd)

    def compute_torque(
        self,
        target_x: float,
        current_x: float,
        current_vx: float,
        current_theta: float,
        current_omega: float,
        current_thrust: float,
        dt: float
    ) -> tuple[float, float]:
        """
        Computes the pitch moment torque command tau [N*m].

        Returns:
            Tuple of (tau_command, desired_theta)
        """
        # Outer Loop: Compute desired horizontal acceleration
        a_x_des = self.pos_pid.compute(
            target=target_x,
            current=current_x,
            dt=dt,
            current_rate=current_vx
        )

        # Invert planar dynamics: x_ddot = - (T / m) * sin(theta)
        # Therefore: sin(theta_des) = - (m * a_x_des) / T
        m = self.params.mass
        effective_thrust = max(current_thrust, 0.4 * self.params.hover_thrust)
        sin_arg = -(m * a_x_des) / effective_thrust

        # Constrain tilt setpoint to safe operating envelope
        max_sin = np.sin(self.params.max_tilt_angle_rad)
        sin_arg_clipped = float(np.clip(sin_arg, -max_sin, max_sin))
        self.desired_theta = float(np.arcsin(sin_arg_clipped))

        # Inner Loop: Regulate pitch attitude to desired tilt
        torque_cmd = self.att_pid.compute(
            target=self.desired_theta,
            current=current_theta,
            dt=dt,
            current_rate=current_omega
        )

        return torque_cmd, self.desired_theta


class CascadedQuadrotorController:
    """
    High-level flight controller orchestrating altitude, position, attitude,
    and operational state transitions.
    """

    def __init__(self, params: DroneParameters | None = None):
        self.params = params if params is not None else DroneParameters()
        self.altitude_ctrl = AltitudePIDController(self.params)
        self.position_ctrl = PositionAttitudeController(self.params)

        self.mode: FlightMode = FlightMode.DISARMED
        self.target_x: float = 0.0
        self.target_z: float = 0.0

        # Landing trajectory ramp parameters
        self.landing_descent_rate: float = 0.8  # m/s

    def reset(self) -> None:
        self.altitude_ctrl.reset()
        self.position_ctrl.reset()
        self.mode = FlightMode.DISARMED
        self.target_x = 0.0
        self.target_z = 0.0

    def command_takeoff(self, target_altitude: float = 5.0) -> None:
        """Initiate autonomous takeoff sequence to target altitude."""
        self.target_z = max(1.0, float(target_altitude))
        self.mode = FlightMode.TAKEOFF

    def command_hover(self, current_x: float, current_z: float) -> None:
        """Lock current spatial coordinate as hold waypoint."""
        self.target_x = float(current_x)
        self.target_z = max(0.5, float(current_z))
        self.mode = FlightMode.HOVER

    def command_move_relative(self, dx: float) -> None:
        """Shift horizontal target waypoint relative to current setpoint."""
        self.target_x += float(dx)
        if self.mode not in (FlightMode.TAKEOFF, FlightMode.LANDING):
            self.mode = FlightMode.TRACKING

    def command_land(self) -> None:
        """Initiate controlled landing descent."""
        self.mode = FlightMode.LANDING

    def command_disarm(self) -> None:
        """Cut motor power immediately."""
        self.mode = FlightMode.DISARMED
        self.target_z = 0.0

    def update(
        self,
        current_x: float,
        current_z: float,
        current_vx: float,
        current_vz: float,
        current_theta: float,
        current_omega: float,
        dt: float
    ) -> tuple[float, float, float]:
        """
        Executes one full control iteration.

        Returns:
            Tuple of (thrust_command, torque_command, desired_pitch_angle)
        """
        # If disarmed or touched down on ground
        if self.mode == FlightMode.DISARMED:
            return 0.0, 0.0, 0.0

        if self.mode == FlightMode.LANDED:
            if current_z <= 0.05:
                return 0.0, 0.0, 0.0
            else:
                self.mode = FlightMode.LANDING

        # Controlled Landing state machine
        if self.mode == FlightMode.LANDING:
            self.target_z = max(0.0, self.target_z - self.landing_descent_rate * dt)
            if current_z <= 0.08 and abs(current_vz) < 0.3:
                self.mode = FlightMode.LANDED
                return 0.0, 0.0, 0.0

        # Autonomous Takeoff state transition check
        if self.mode == FlightMode.TAKEOFF:
            if abs(current_z - self.target_z) < 0.25 and abs(current_vz) < 0.2:
                self.mode = FlightMode.HOVER

        # Altitude loop (compute thrust)
        thrust_cmd = self.altitude_ctrl.compute_thrust(
            target_z=self.target_z,
            current_z=current_z,
            current_vz=current_vz,
            pitch_theta=current_theta,
            dt=dt
        )

        # Position and Attitude loops (compute torque)
        torque_cmd, desired_theta = self.position_ctrl.compute_torque(
            target_x=self.target_x,
            current_x=current_x,
            current_vx=current_vx,
            current_theta=current_theta,
            current_omega=current_omega,
            current_thrust=thrust_cmd,
            dt=dt
        )

        return thrust_cmd, torque_cmd, desired_theta

