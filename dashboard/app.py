"""
Streamlit Aerospace Telemetry & Control Dashboard.
Intelligent Systems and Control (ISC) Project.

Features:
- Dark theme aerospace telemetry HUD
- Real-time parameter tuning (PID gains, drone physical parameters)
- Action buttons for Takeoff, Land, Hover, Translation, and Wind injection
- Live multi-panel performance visualization
- Genuine real-time metric calculations (MAE, RMSE, Overshoot, Settling Time, Steady-State Error)
- Benchmark test scenario runner
"""

import sys
import os
# Ensure root workspace directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drone.parameters import DroneParameters
from simulation.simulator import QuadrotorSimulator
from controllers.position_control import FlightMode
from evaluation.metrics import calculate_flight_metrics
from evaluation.plots import generate_all_plots


# ---------------- PAGE CONFIG & DARK THEME CSS ----------------
st.set_page_config(
    page_title="Quadrotor Intelligent Control Telemetry",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Dark Theme Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Aerospace Telemetry Card */
    .telemetry-card {
        background: linear-gradient(135deg, #131c2e 0%, #172338 100%);
        border: 1px solid #24344d;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .telemetry-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8da2c0;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .telemetry-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .telemetry-sub {
        font-size: 12px;
        color: #00d2ff;
        margin-top: 4px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .status-hover { background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .status-takeoff { background-color: rgba(52, 152, 219, 0.2); color: #3498db; border: 1px solid #3498db; }
    .status-landing { background-color: rgba(230, 126, 34, 0.2); color: #e67e22; border: 1px solid #e67e22; }
    .status-tracking { background-color: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }
    .status-disarmed { background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }

    /* Header styling */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE INITIALIZATION ----------------
if "simulator" not in st.session_state:
    params = DroneParameters()
    sim = QuadrotorSimulator(params)
    # Warm up with a default takeoff flight so graphs are populated
    sim.controller.command_takeoff(target_altitude=5.0)
    for _ in range(400):  # 4 seconds at 100Hz
        sim.step(0.01)
    sim.controller.command_hover(0.0, 5.0)
    for _ in range(300):
        sim.step(0.01)
    st.session_state.simulator = sim

sim: QuadrotorSimulator = st.session_state.simulator


# ---------------- SIDEBAR CONTROLS & GAIN TUNING ----------------
st.sidebar.image("https://img.icons8.com/fluency/96/drone.png", width=64)
st.sidebar.title("Flight Avionics Control")
st.sidebar.caption("Intelligent Systems & Control (ISC)")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Flight Waypoints")
target_z_input = st.sidebar.slider("Target Altitude Z [m]", min_value=0.5, max_value=10.0, value=float(sim.controller.target_z or 5.0), step=0.5)
target_x_input = st.sidebar.slider("Target Horizontal X [m]", min_value=-10.0, max_value=10.0, value=float(sim.controller.target_x or 0.0), step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Atmospheric Disturbance")
wind_toggle = st.sidebar.checkbox("Enable Wind Disturbance", value=sim.dynamics.wind.enabled)
wind_strength = st.sidebar.slider("Base Wind Force [N]", min_value=-5.0, max_value=5.0, value=float(sim.dynamics.wind.base_force_x or 2.0), step=0.5)
gust_amp = st.sidebar.slider("Gust Amplitude [N]", min_value=0.0, max_value=5.0, value=float(sim.dynamics.wind.gust_amplitude), step=0.5)

sim.dynamics.wind.enabled = wind_toggle
sim.dynamics.wind.base_force_x = wind_strength
sim.dynamics.wind.gust_amplitude = gust_amp

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cascaded PID Tuner")

with st.sidebar.expander("Vertical Altitude PID Gains", expanded=False):
    alt_kp = st.slider("Altitude Kp", 1.0, 30.0, float(sim.controller.altitude_ctrl.pid.kp), 0.5)
    alt_ki = st.slider("Altitude Ki", 0.0, 10.0, float(sim.controller.altitude_ctrl.pid.ki), 0.1)
    alt_kd = st.slider("Altitude Kd", 0.5, 20.0, float(sim.controller.altitude_ctrl.pid.kd), 0.5)
    sim.controller.altitude_ctrl.update_gains(alt_kp, alt_ki, alt_kd)

with st.sidebar.expander("Outer Position PID Gains", expanded=False):
    pos_kp = st.slider("Position Kp", 0.2, 10.0, float(sim.controller.position_ctrl.pos_pid.kp), 0.1)
    pos_ki = st.slider("Position Ki", 0.0, 2.0, float(sim.controller.position_ctrl.pos_pid.ki), 0.05)
    pos_kd = st.slider("Position Kd", 0.2, 10.0, float(sim.controller.position_ctrl.pos_pid.kd), 0.1)
    sim.controller.position_ctrl.pos_pid.kp = pos_kp
    sim.controller.position_ctrl.pos_pid.ki = pos_ki
    sim.controller.position_ctrl.pos_pid.kd = pos_kd

with st.sidebar.expander("Inner Attitude PID Gains", expanded=False):
    att_kp = st.slider("Attitude Kp", 1.0, 30.0, float(sim.controller.position_ctrl.att_pid.kp), 0.5)
    att_ki = st.slider("Attitude Ki", 0.0, 5.0, float(sim.controller.position_ctrl.att_pid.ki), 0.1)
    att_kd = st.slider("Attitude Kd", 0.2, 10.0, float(sim.controller.position_ctrl.att_pid.kd), 0.1)
    sim.controller.position_ctrl.att_pid.kp = att_kp
    sim.controller.position_ctrl.att_pid.ki = att_ki
    sim.controller.position_ctrl.att_pid.kd = att_kd

# ---------------- TOP COMMAND BAR ----------------
st.title("🚁 Real-Time Quadrotor Intelligent Control Telemetry")
st.markdown("**Autonomous Flight • Cascaded PID Feedback • Aerodynamic Disturbance Rejection**")

# Flight Command Action Buttons
b_col1, b_col2, b_col3, b_col4, b_col5, b_col6, b_col7, b_col8 = st.columns(8)

with b_col1:
    if st.button("🚀 Takeoff", use_container_width=True):
        sim.controller.command_takeoff(target_z_input)
        for _ in range(150):
            sim.step(0.01)
        st.rerun()

with b_col2:
    if st.button("⏸️ Hover", use_container_width=True):
        sim.controller.command_hover(sim.dynamics.x, sim.dynamics.z)
        for _ in range(100):
            sim.step(0.01)
        st.rerun()

with b_col3:
    if st.button("⬅️ Left (-2m)", use_container_width=True):
        sim.controller.command_move_relative(-2.0)
        for _ in range(150):
            sim.step(0.01)
        st.rerun()

with b_col4:
    if st.button("➡️ Right (+2m)", use_container_width=True):
        sim.controller.command_move_relative(2.0)
        for _ in range(150):
            sim.step(0.01)
        st.rerun()

with b_col5:
    if st.button("🛬 Land", use_container_width=True):
        sim.controller.command_land()
        for _ in range(250):
            sim.step(0.01)
        st.rerun()

with b_col6:
    if st.button("💨 Wind Gust", use_container_width=True):
        sim.toggle_wind(3.0)
        for _ in range(150):
            sim.step(0.01)
        st.rerun()

with b_col7:
    if st.button("🔄 Step +1s", use_container_width=True):
        for _ in range(100):
            sim.step(0.01)
        st.rerun()

with b_col8:
    if st.button("⚠️ Reset", use_container_width=True):
        sim.reset()
        st.rerun()

st.markdown("---")

# ---------------- LIVE TELEMETRY CARDS ----------------
# Extract latest state
last = sim.history[-1] if len(sim.history) > 0 else sim.step(0.01)
df = sim.get_history_df()

mode_str = last["mode"]
badge_class = f"status-{mode_str.lower()}" if mode_str.lower() in ["hover", "takeoff", "landing", "tracking"] else "status-disarmed"

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Flight Mode</div>
        <div style="margin-top: 6px;"><span class="status-badge {badge_class}">{mode_str}</span></div>
        <div class="telemetry-sub">Sim Time: {last['time']:.2f} s</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Altitude (Z)</div>
        <div class="telemetry-value">{last['z']:.2f} <span style="font-size:16px;">m</span></div>
        <div class="telemetry-sub">Target: {last['target_z']:.1f} m  (Δ {last['z'] - last['target_z']:+.2f}m)</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Position (X)</div>
        <div class="telemetry-value">{last['x']:.2f} <span style="font-size:16px;">m</span></div>
        <div class="telemetry-sub">Target: {last['target_x']:.1f} m  (Δ {last['x'] - last['target_x']:+.2f}m)</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Vertical Vel (Vz)</div>
        <div class="telemetry-value">{last['vz']:+.2f} <span style="font-size:16px;">m/s</span></div>
        <div class="telemetry-sub">Horiz: {last['vx']:+.2f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

with m_col5:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Pitch Angle (θ)</div>
        <div class="telemetry-value">{last['theta_deg']:+.2f}°</div>
        <div class="telemetry-sub">Desired: {last['desired_theta_deg']:+.2f}° | ω: {last['omega_deg']:+.1f}°/s</div>
    </div>
    """, unsafe_allow_html=True)

with m_col6:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">Total Thrust (T)</div>
        <div class="telemetry-value">{last['thrust']:.2f} <span style="font-size:16px;">N</span></div>
        <div class="telemetry-sub">{last['thrust_pct']:.1f}% Max | τ: {last['torque']:+.3f} N*m</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MULTI-TAB VISUALIZATION & BENCHMARKS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trajectory & Tracking",
    "⚙️ Control & Actuators",
    "🎯 2D Flight Profile",
    "📊 Quantitative Performance Metrics",
    "🧪 ISC Benchmark Scenarios"
])

# Styling for Matplotlib charts in Streamlit
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor": "#1F2937",
    "axes.edgecolor": "#374151",
    "axes.labelcolor": "#F3F4F6",
    "axes.grid": True,
    "grid.color": "#374151",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "xtick.color": "#9CA3AF",
    "ytick.color": "#9CA3AF",
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#374151",
    "text.color": "#F3F4F6",
    "font.family": "sans-serif",
    "font.size": 10
})

with tab1:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Altitude Tracking (Desired vs Actual)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["time"], df["target_z"], "r--", linewidth=1.8, label="Desired Z")
        ax.plot(df["time"], df["z"], "#00D2FF", linewidth=2.0, label="Actual Z")
        ax.set_ylabel("Altitude [m]")
        ax.set_xlabel("Time [s]")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_t2:
        st.subheader("Horizontal Position (Desired vs Actual)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["time"], df["target_x"], "r--", linewidth=1.8, label="Desired X")
        ax.plot(df["time"], df["x"], "#10B981", linewidth=2.0, label="Actual X")
        ax.set_ylabel("Position X [m]")
        ax.set_xlabel("Time [s]")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Inner-Loop Attitude (Pitch Angle)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["time"], df["desired_theta_deg"], "r--", linewidth=1.5, label="Desired θ")
        ax.plot(df["time"], df["theta_deg"], "#C084FC", linewidth=2.0, label="Actual Pitch θ")
        ax.axhline(30.0, color="#EF4444", linestyle=":", label="Limit (±30°)")
        ax.axhline(-30.0, color="#EF4444", linestyle=":")
        ax.set_ylabel("Pitch Angle [deg]")
        ax.set_xlabel("Time [s]")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_c2:
        st.subheader("Actuation Commands (Thrust & Moment)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["time"], df["thrust"], "#F59E0B", linewidth=2.0, label="Thrust T [N]")
        ax.axhline(sim.params.hover_thrust, color="#60A5FA", linestyle="--", label="Hover T_hover")
        ax.set_ylabel("Thrust [N]")
        ax.set_xlabel("Time [s]")
        
        ax_tw = ax.twinx()
        ax_tw.plot(df["time"], df["torque"], "#38BDF8", linewidth=1.5, label="Torque τ [N*m]", alpha=0.8)
        ax_tw.set_ylabel("Torque [N*m]", color="#38BDF8")
        ax.legend(loc="upper left")
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("2D Vertical Flight Profile (X vs Z)")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["x"], df["z"], "#3B82F6", linewidth=2.2, label="Flight Trajectory")
    ax.scatter([df["x"].iloc[0]], [df["z"].iloc[0]], color="#10B981", s=80, label="Origin / Liftoff", zorder=5)
    ax.scatter([df["x"].iloc[-1]], [df["z"].iloc[-1]], color="#EF4444", s=80, label="Current Drone Position", zorder=5)
    ax.axhline(0, color="#64748B", linewidth=2.5, label="Ground Surface (z=0)")
    ax.set_xlabel("Horizontal Coordinate X [m]")
    ax.set_ylabel("Altitude Coordinate Z [m]")
    ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("📊 Rigorous Academic Control Performance Metrics")
    metrics = calculate_flight_metrics(df)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("### Altitude Control Metrics")
        alt_m = metrics.get("altitude", {})
        st.write(f"- **Mean Absolute Error (MAE):** `{alt_m.get('MAE_m', 'N/A')} m`")
        st.write(f"- **Root Mean Square Error (RMSE):** `{alt_m.get('RMSE_m', 'N/A')} m`")
        st.write(f"- **Maximum Overshoot:** `{alt_m.get('max_overshoot_pct', 'N/A')}%`")
        st.write(f"- **Settling Time (±5%):** `{alt_m.get('settling_time_s', 'N/A')} s`")
        st.write(f"- **Steady-State Error (Ess):** `{alt_m.get('steady_state_error_m', 'N/A')} m`")

    with col_m2:
        st.markdown("### Horizontal Position Metrics")
        pos_m = metrics.get("position", {})
        st.write(f"- **Mean Absolute Error (MAE):** `{pos_m.get('MAE_m', 'N/A')} m`")
        st.write(f"- **Root Mean Square Error (RMSE):** `{pos_m.get('RMSE_m', 'N/A')} m`")
        st.write(f"- **Steady-State Error (Ess):** `{pos_m.get('steady_state_error_m', 'N/A')} m`")
        st.write(f"- **Max Displacement:** `{pos_m.get('max_x_displacement_m', 'N/A')} m`")

    with col_m3:
        st.markdown("### Attitude & Control Metrics")
        att_m = metrics.get("attitude", {})
        eff_m = metrics.get("control_effort", {})
        st.write(f"- **Attitude MAE:** `{att_m.get('MAE_deg', 'N/A')}°`")
        st.write(f"- **Attitude RMSE:** `{att_m.get('RMSE_deg', 'N/A')}°`")
        st.write(f"- **Mean Actuation Thrust:** `{eff_m.get('mean_thrust_N', 'N/A')} N`")
        st.write(f"- **Max Actuation Thrust:** `{eff_m.get('max_thrust_N', 'N/A')} N`")
        st.write(f"- **Mean Pitch Moment:** `{eff_m.get('mean_abs_torque_Nm', 'N/A')} N*m`")

with tab5:
    st.subheader("🧪 Run ISC Validation Scenarios")
    st.markdown("Execute standard academic Intelligent Systems and Control (ISC) scenarios and generate evaluation figures.")

    scenario_selected = st.selectbox(
        "Choose Validation Scenario:",
        [
            "TEST 1: Autonomous Takeoff & Altitude Stabilization",
            "TEST 2: Stationary Hovering Flight",
            "TEST 3: Multi-Waypoint Step Translation (Left & Right)",
            "TEST 4: Autonomous Controlled Landing",
            "TEST 5: Aerodynamic Wind Disturbance Rejection",
            "TEST 6: Comprehensive Benchmark Suite (All Tests)"
        ]
    )

    if st.button("▶️ Execute Benchmark Scenario"):
        with st.spinner("Simulating scenario and computing physical dynamics..."):
            test_sim = QuadrotorSimulator()
            if "TEST 1" in scenario_selected:
                test_sim.controller.command_takeoff(5.0)
                for _ in range(700):
                    test_sim.step(0.01)
            elif "TEST 2" in scenario_selected:
                test_sim.controller.command_takeoff(4.0)
                for _ in range(400):
                    test_sim.step(0.01)
                test_sim.controller.command_hover(0.0, 4.0)
                for _ in range(600):
                    test_sim.step(0.01)
            elif "TEST 3" in scenario_selected:
                test_sim.controller.command_takeoff(4.0)
                for _ in range(300):
                    test_sim.step(0.01)
                test_sim.controller.command_move_relative(3.0)
                for _ in range(400):
                    test_sim.step(0.01)
                test_sim.controller.command_move_relative(-6.0)
                for _ in range(500):
                    test_sim.step(0.01)
            elif "TEST 4" in scenario_selected:
                test_sim.controller.command_takeoff(4.0)
                for _ in range(400):
                    test_sim.step(0.01)
                test_sim.controller.command_land()
                for _ in range(700):
                    test_sim.step(0.01)
            elif "TEST 5" in scenario_selected:
                test_sim.controller.command_takeoff(5.0)
                for _ in range(300):
                    test_sim.step(0.01)
                # Inject wind disturbance
                test_sim.toggle_wind(base_force=3.0)
                for _ in range(400):
                    test_sim.step(0.01)
                test_sim.toggle_wind()
                for _ in range(300):
                    test_sim.step(0.01)
            else:
                # Comprehensive Suite
                test_sim.controller.command_takeoff(5.0)
                for _ in range(400):
                    test_sim.step(0.01)
                test_sim.controller.command_move_relative(3.0)
                for _ in range(350):
                    test_sim.step(0.01)
                test_sim.toggle_wind(2.5)
                for _ in range(350):
                    test_sim.step(0.01)
                test_sim.toggle_wind()
                for _ in range(200):
                    test_sim.step(0.01)
                test_sim.controller.command_land()
                for _ in range(600):
                    test_sim.step(0.01)

            st.session_state.simulator = test_sim
            os.makedirs("results", exist_ok=True)
            res_df = test_sim.get_history_df()
            generate_all_plots(res_df, "results")
            st.success("Scenario completed! Saved performance plots to 'results/'. Updating dashboard...")
            st.rerun()

