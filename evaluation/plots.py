"""
Publication-Quality Performance Plotting for Control Evaluation.
Intelligent Systems and Control (ISC) Project.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all platforms
import matplotlib.pyplot as plt


def apply_engineering_style():
    """Applies modern engineering styling to matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0F141C",
        "axes.facecolor": "#171F2C",
        "axes.edgecolor": "#2D3E54",
        "axes.labelcolor": "#E0E6ED",
        "axes.grid": True,
        "grid.color": "#233042",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "xtick.color": "#B0C0D0",
        "ytick.color": "#B0C0D0",
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#334155",
        "text.color": "#FFFFFF",
        "font.family": "sans-serif",
        "font.size": 11,
    })


def plot_altitude_response(df: pd.DataFrame, output_path: str) -> None:
    """1. Desired Altitude vs Actual Altitude and Error."""
    apply_engineering_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    ax1.plot(df["time"], df["target_z"], "r--", linewidth=2.0, label="Desired Altitude (Z_ref)", alpha=0.9)
    ax1.plot(df["time"], df["z"], "#00D2FF", linewidth=2.2, label="Actual Altitude (Z)")
    ax1.set_ylabel("Altitude [m]")
    ax1.set_title("Altitude Stabilization & Tracking Response (Cascaded PID)", fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc="lower right")

    # Error subpanel
    error_z = df["target_z"] - df["z"]
    ax2.plot(df["time"], error_z, "#FF7675", linewidth=1.5, label="Tracking Error (e_z)")
    ax2.axhline(0, color="#636E72", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [m]")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_position_response(df: pd.DataFrame, output_path: str) -> None:
    """2. Desired Horizontal Position vs Actual Position."""
    apply_engineering_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    ax1.plot(df["time"], df["target_x"], "r--", linewidth=2.0, label="Desired Position (X_ref)", alpha=0.9)
    ax1.plot(df["time"], df["x"], "#2ECC71", linewidth=2.2, label="Actual Position (X)")
    ax1.set_ylabel("Position X [m]")
    ax1.set_title("Horizontal Position Tracking & Hovering Performance", fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc="lower right")

    # Error subpanel
    error_x = df["target_x"] - df["x"]
    ax2.plot(df["time"], error_x, "#F39C12", linewidth=1.5, label="Position Error (e_x)")
    ax2.axhline(0, color="#636E72", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [m]")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pitch_angle(df: pd.DataFrame, output_path: str) -> None:
    """3. Pitch Angle Over Time."""
    apply_engineering_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["time"], df["desired_theta_deg"], "r--", linewidth=1.8, label="Desired Pitch (θ_des)", alpha=0.85)
    ax.plot(df["time"], df["theta_deg"], "#E056FD", linewidth=2.2, label="Actual Pitch (θ)")
    ax.axhline(30.0, color="#FF4757", linestyle=":", label="Pitch Limit (+30°)")
    ax.axhline(-30.0, color="#FF4757", linestyle=":", label="Pitch Limit (-30°)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pitch Angle [deg]")
    ax.set_title("Inner-Loop Attitude (Pitch) Dynamics and Tracking", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_control_output(df: pd.DataFrame, output_path: str) -> None:
    """4. Control Outputs: Total Thrust and Rotational Torque over time."""
    apply_engineering_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Thrust subplot
    ax1.plot(df["time"], df["thrust"], "#FF9F43", linewidth=2.0, label="Commanded Thrust (T)")
    ax1.axhline(1.2 * 9.81, color="#74B9FF", linestyle="--", linewidth=1.5, label="Hover Thrust (m*g)")
    ax1.axhline(24.0, color="#EE5253", linestyle=":", label="Max Thrust Saturation (24 N)")
    ax1.set_ylabel("Thrust [N]")
    ax1.set_title("Actuator Control Outputs and Saturation Envelope", fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc="upper right")

    # Torque subplot
    ax2.plot(df["time"], df["torque"], "#54A0FF", linewidth=1.8, label="Pitch Torque (τ)")
    ax2.axhline(2.5, color="#EE5253", linestyle=":", label="Max Torque (±2.5 N*m)")
    ax2.axhline(-2.5, color="#EE5253", linestyle=":")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Torque [N*m]")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_disturbance_rejection(df: pd.DataFrame, output_path: str) -> None:
    """5. Flight Response During External Wind Disturbance."""
    apply_engineering_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Wind force profile
    ax1.plot(df["time"], df["wind_fx"], "#FF6B6B", linewidth=2.0, label="External Wind Disturbance Force (Fx_wind)")
    ax1.set_ylabel("Wind Force [N]")
    ax1.set_title("Disturbance Rejection: Aerodynamic Wind Gust & Feedback Recovery", fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc="upper right")

    # Vehicle response
    ax2.plot(df["time"], df["target_x"], "r--", linewidth=1.8, label="Target X Position", alpha=0.8)
    ax2.plot(df["time"], df["x"], "#1DD1A1", linewidth=2.2, label="Actual X Position (Correcting)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position X [m]")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_trajectory_2d(df: pd.DataFrame, output_path: str) -> None:
    """6. 2D Spatial Trajectory (X vs Z)."""
    apply_engineering_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(df["x"], df["z"], "#0984E3", linewidth=2.2, label="Drone 2D Flight Path")
    ax.scatter([df["x"].iloc[0]], [df["z"].iloc[0]], color="#00B894", s=90, zorder=5, label="Takeoff Start")
    ax.scatter([df["x"].iloc[-1]], [df["z"].iloc[-1]], color="#D63031", s=90, zorder=5, label="Final Location")

    # Unique target waypoints
    targets = df[["target_x", "target_z"]].drop_duplicates()
    ax.scatter(targets["target_x"], targets["target_z"], color="#FDCB6E", marker="X", s=110, zorder=4, label="Target Setpoints")

    ax.axhline(0, color="#636E72", linewidth=2.5, label="Ground Surface (z=0)")
    ax.set_xlabel("Horizontal Position X [m]")
    ax.set_ylabel("Altitude Z [m]")
    ax.set_title("2D Spatial Trajectory in Vertical Flight Plane", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(-0.5, max(8.0, df["z"].max() + 1.5))
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_consolidated_summary(df: pd.DataFrame, metrics: Dict[str, Any], output_path: str) -> None:
    """7. Consolidated Multi-Panel Presentation Poster."""
    apply_engineering_style()
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Altitude
    axs[0, 0].plot(df["time"], df["target_z"], "r--", label="Target Z")
    axs[0, 0].plot(df["time"], df["z"], "#00D2FF", label="Actual Z")
    axs[0, 0].set_title("Altitude Tracking [m]", fontweight="bold")
    axs[0, 0].set_ylabel("Altitude [m]")
    axs[0, 0].legend()

    # Panel 2: Position
    axs[0, 1].plot(df["time"], df["target_x"], "r--", label="Target X")
    axs[0, 1].plot(df["time"], df["x"], "#2ECC71", label="Actual X")
    axs[0, 1].set_title("Horizontal Position Tracking [m]", fontweight="bold")
    axs[0, 1].set_ylabel("Position [m]")
    axs[0, 1].legend()

    # Panel 3: Pitch Attitude
    axs[1, 0].plot(df["time"], df["desired_theta_deg"], "r--", label="Desired θ")
    axs[1, 0].plot(df["time"], df["theta_deg"], "#E056FD", label="Actual θ")
    axs[1, 0].set_title("Pitch Attitude Response [deg]", fontweight="bold")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Angle [deg]")
    axs[1, 0].legend()

    # Panel 4: Actuator Effort
    axs[1, 1].plot(df["time"], df["thrust"], "#FF9F43", label="Thrust [N]")
    ax_twin = axs[1, 1].twinx()
    ax_twin.plot(df["time"], df["torque"], "#54A0FF", label="Torque [N*m]", alpha=0.85)
    ax_twin.set_ylabel("Torque [N*m]", color="#54A0FF")
    axs[1, 1].set_title("Control Actuation (Thrust & Moment)", fontweight="bold")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Thrust [N]")
    axs[1, 1].legend(loc="upper left")

    plt.suptitle("ISC Quadrotor Intelligent Control System - Performance Summary", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, output_dir: str = "results") -> None:
    """Batch generates all 6 core required performance plots + summaries."""
    os.makedirs(output_dir, exist_ok=True)
    from .metrics import calculate_flight_metrics

    metrics = calculate_flight_metrics(df)

    plot_altitude_response(df, os.path.join(output_dir, "desired_vs_actual_altitude.png"))
    plot_position_response(df, os.path.join(output_dir, "desired_vs_actual_position.png"))
    plot_pitch_angle(df, os.path.join(output_dir, "pitch_angle_vs_time.png"))
    plot_control_output(df, os.path.join(output_dir, "control_output_vs_time.png"))
    plot_disturbance_rejection(df, os.path.join(output_dir, "flight_response_wind_disturbance.png"))
    plot_trajectory_2d(df, os.path.join(output_dir, "flight_trajectory_2d.png"))
    plot_consolidated_summary(df, metrics, os.path.join(output_dir, "isc_benchmark_summary.png"))

