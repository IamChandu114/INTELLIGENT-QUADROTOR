"""
Robust Proportional-Integral-Derivative (PID) Controller with Anti-Windup.
Intelligent Systems and Control (ISC) Project.
"""

from typing import Tuple
import numpy as np


class PIDController:
    """
    Standard industrial/aerospace PID controller with:
      - Configurable gains (Kp, Ki, Kd)
      - Anti-windup integration clamping
      - Derivative low-pass filtering to suppress high-frequency noise
      - Output saturation limits
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -np.inf,
        output_max: float = np.inf,
        integrator_min: float = -np.inf,
        integrator_max: float = np.inf,
        derivative_filter_alpha: float = 0.8,
        name: str = "PID"
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.integrator_min = float(integrator_min)
        self.integrator_max = float(integrator_max)
        self.derivative_filter_alpha = float(derivative_filter_alpha)
        self.name = name

        # State memory
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.prev_derivative: float = 0.0
        self.prev_measurement: float | None = None
        self.has_run: bool = False

        # Diagnostic breakdown of last step
        self.p_term: float = 0.0
        self.i_term: float = 0.0
        self.d_term: float = 0.0
        self.last_output: float = 0.0

    def reset(self) -> None:
        """Clear historical integrator and derivative terms."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_measurement = None
        self.has_run = False
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.last_output = 0.0

    def compute(
        self,
        target: float,
        current: float,
        dt: float,
        feedforward: float = 0.0,
        current_rate: float | None = None
    ) -> float:
        """
        Calculates corrective control effort u(t).

        Args:
            target: Desired reference setpoint r(t)
            current: Current measured process variable y(t)
            dt: Simulation sample period [s]
            feedforward: Feedforward compensation term
            current_rate: Optional direct rate measurement (dy/dt) to bypass numeric derivative

        Returns:
            Saturated control output u(t)
        """
        if dt <= 0.0:
            return 0.0

        error = target - current

        # Proportional term
        self.p_term = self.kp * error

        # Derivative term
        if current_rate is not None:
            # Direct derivative on process variable (negative feedback)
            derivative = -current_rate
        elif self.has_run:
            raw_derivative = (error - self.prev_error) / dt
            # Low-pass filter derivative to reject numeric jitter: y_d = alpha * y_prev + (1 - alpha) * raw
            derivative = (
                self.derivative_filter_alpha * self.prev_derivative
                + (1.0 - self.derivative_filter_alpha) * raw_derivative
            )
        else:
            derivative = 0.0

        self.d_term = self.kd * derivative
        self.prev_derivative = derivative
        self.prev_error = error

        # Tentative output before integrator update for clamping check
        tentative_output = self.p_term + self.ki * self.integral + self.d_term + feedforward

        # Integral term with Anti-Windup Protection (Conditional Integration / Clamping)
        # Only integrate if output is not saturated, OR if the integration acts to pull out of saturation
        will_saturate_high = tentative_output >= self.output_max and error > 0.0
        will_saturate_low = tentative_output <= self.output_min and error < 0.0

        if not (will_saturate_high or will_saturate_low):
            self.integral += error * dt
            # Constrain integral within specified safety bounds
            self.integral = float(np.clip(self.integral, self.integrator_min, self.integrator_max))

        self.i_term = self.ki * self.integral

        # Total unconstrained command
        raw_output = self.p_term + self.i_term + self.d_term + feedforward

        # Actuator Saturation Limit
        self.last_output = float(np.clip(raw_output, self.output_min, self.output_max))
        self.has_run = True

        return self.last_output

