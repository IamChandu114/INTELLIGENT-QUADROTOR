"""
Quantitative Flight Performance Evaluation Metrics.
Intelligent Systems and Control (ISC) Project.

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum Percentage Overshoot (Mp)
- Settling Time (Ts) to within tolerance band
- Steady-State Error (Ess)
- Disturbance Rejection Recovery Time
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def calculate_flight_metrics(df: pd.DataFrame, tolerance_pct: float = 0.05) -> Dict[str, Any]:
    """
    Computes rigorous control metrics on simulation time-series data.
    All calculations derive from genuine recorded state logs.
    """
    if df.empty or len(df) < 5:
        return {}

    # Altitude Metrics
    err_z = df["target_z"].values - df["z"].values
    mae_z = float(np.mean(np.abs(err_z)))
    rmse_z = float(np.sqrt(np.mean(err_z ** 2)))

    # Steady-state error over final 15% of simulation
    steady_slice_len = max(5, int(len(df) * 0.15))
    final_err_z = err_z[-steady_slice_len:]
    ess_z = float(np.mean(np.abs(final_err_z)))

    # Step response metrics for altitude (takeoff phase)
    # Target altitude step
    max_target_z = float(np.max(df["target_z"]))
    max_z = float(np.max(df["z"]))
    if max_target_z > 0.5:
        overshoot_z_pct = max(0.0, float((max_z - max_target_z) / max_target_z * 100.0))
    else:
        overshoot_z_pct = 0.0

    # Settling time for altitude (time to stay within +- tolerance band of target_z)
    settling_time_z = None
    if max_target_z > 0.5:
        band = tolerance_pct * max_target_z
        # Find the last time the absolute error was outside the band
        outside_indices = np.where(np.abs(err_z) > band)[0]
        if len(outside_indices) == 0:
            settling_time_z = 0.0
        elif outside_indices[-1] < len(df) - 1:
            settling_time_z = float(df["time"].iloc[outside_indices[-1] + 1])
        else:
            settling_time_z = float(df["time"].iloc[-1])

    # Horizontal Position Metrics
    err_x = df["target_x"].values - df["x"].values
    mae_x = float(np.mean(np.abs(err_x)))
    rmse_x = float(np.sqrt(np.mean(err_x ** 2)))
    final_err_x = err_x[-steady_slice_len:]
    ess_x = float(np.mean(np.abs(final_err_x)))

    # Attitude Tracking Metrics
    err_theta = df["desired_theta_deg"].values - df["theta_deg"].values
    mae_theta = float(np.mean(np.abs(err_theta)))
    rmse_theta = float(np.sqrt(np.mean(err_theta ** 2)))

    # Disturbance Recovery Metric
    # If wind was enabled, calculate max displacement and recovery
    wind_rows = df[df["wind_enabled"] == True]
    dist_metrics = {}
    if not wind_rows.empty:
        max_wind_x_dev = float(np.max(np.abs(wind_rows["x"].values - wind_rows["target_x"].values)))
        dist_metrics["max_wind_deviation_m"] = round(max_wind_x_dev, 4)
        dist_metrics["wind_sample_count"] = int(len(wind_rows))

    return {
        "altitude": {
            "MAE_m": round(mae_z, 4),
            "RMSE_m": round(rmse_z, 4),
            "max_overshoot_pct": round(overshoot_z_pct, 2),
            "settling_time_s": round(settling_time_z, 3) if settling_time_z is not None else "N/A",
            "steady_state_error_m": round(ess_z, 4),
            "max_altitude_m": round(max_z, 3),
        },
        "position": {
            "MAE_m": round(mae_x, 4),
            "RMSE_m": round(rmse_x, 4),
            "steady_state_error_m": round(ess_x, 4),
            "max_x_displacement_m": round(float(np.max(np.abs(df["x"]))), 3),
        },
        "attitude": {
            "MAE_deg": round(mae_theta, 4),
            "RMSE_deg": round(rmse_theta, 4),
            "max_pitch_deg": round(float(np.max(np.abs(df["theta_deg"]))), 2),
        },
        "control_effort": {
            "mean_thrust_N": round(float(np.mean(df["thrust"])), 3),
            "max_thrust_N": round(float(np.max(df["thrust"])), 3),
            "mean_abs_torque_Nm": round(float(np.mean(np.abs(df["torque"]))), 4),
            "max_abs_torque_Nm": round(float(np.max(np.abs(df["torque"]))), 4),
        },
        "disturbance": dist_metrics
    }


def calculate_step_response_metrics(
    time_arr: np.ndarray,
    response_arr: np.ndarray,
    target_value: float,
    tolerance: float = 0.05
) -> Dict[str, float]:
    """Computes standard control system step response metrics."""
    err = np.abs(response_arr - target_value)
    band = tolerance * abs(target_value) if abs(target_value) > 1e-3 else tolerance

    # Maximum overshoot
    peak = np.max(response_arr)
    overshoot = max(0.0, (peak - target_value) / target_value * 100.0) if target_value > 0 else 0.0

    # Settling time
    outside = np.where(err > band)[0]
    ts = float(time_arr[outside[-1] + 1]) if len(outside) > 0 and outside[-1] < len(time_arr) - 1 else float(time_arr[-1])

    # Rise time (10% to 90%)
    t_10_idx = np.where(response_arr >= 0.10 * target_value)[0]
    t_90_idx = np.where(response_arr >= 0.90 * target_value)[0]
    if len(t_10_idx) > 0 and len(t_90_idx) > 0:
        rise_time = float(time_arr[t_90_idx[0]] - time_arr[t_10_idx[0]])
    else:
        rise_time = 0.0

    return {
        "overshoot_pct": round(overshoot, 2),
        "settling_time_s": round(ts, 3),
        "rise_time_s": round(rise_time, 3),
        "steady_state_error": round(float(err[-1]), 4)
    }

