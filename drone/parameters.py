"""
Drone Physical Parameters Specification.
Intelligent Systems and Control (ISC) Project.

Defines the physical characteristics, actuation limits, and environment constants
for a planar 2D quadrotor model.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DroneParameters:
    """Physical and operational parameters of the quadrotor."""

    # Mass and Inertial Properties
    mass: float = 1.2                  # Total vehicle mass [kg]
    arm_length: float = 0.25           # Distance from center of mass to rotor axis [m]
    I_yy: float = 0.025                # Pitch moment of inertia [kg * m^2]
    
    # Gravitational and Environmental Constants
    gravity: float = 9.81              # Gravitational acceleration [m/s^2]
    air_density: float = 1.225         # Sea-level standard air density [kg/m^3]
    rotational_damping: float = 0.05   # Aerodynamic angular damping coefficient [N*m*s/rad]
    linear_drag_x: float = 0.15        # Translational drag coefficient X [N*s/m]
    linear_drag_z: float = 0.15        # Translational drag coefficient Z [N*s/m]

    # Actuation Limits
    thrust_min: float = 0.0            # Minimum total motor thrust [N]
    thrust_max: float = 24.0           # Maximum total motor thrust [N] (T/W ratio ~ 2.04)
    torque_max: float = 2.5            # Maximum rotational pitch torque [N*m]
    
    # Kinematic Safety Constraints
    max_tilt_angle_deg: float = 30.0   # Max allowable pitch angle [degrees]
    max_tilt_angle_rad: float = np.radians(30.0) # Max pitch angle [radians]
    max_horizontal_velocity: float = 4.0 # Max safe horizontal speed [m/s]
    max_vertical_velocity: float = 3.0   # Max safe vertical speed [m/s]
    max_angular_velocity: float = 5.0    # Max safe angular velocity [rad/s]

    # Simulation Temporal Configuration
    dt: float = 0.01                   # Physics integration step size [s] (100 Hz)

    @property
    def hover_thrust(self) -> float:
        """Thrust required to maintain steady-state hover: T_hover = m * g."""
        return self.mass * self.gravity

