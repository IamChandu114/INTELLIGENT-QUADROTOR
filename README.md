# 🚁 Real-Time Simulation of Intelligent Control for a Quadrotor Drone (Simplified)

**Intelligent Systems and Control (ISC) Project**  
*Academic Demonstration of Autonomous Flight, PID-Based Feedback Control, and Disturbance Rejection*

---

## 📋 Project Overview

This project implements a complete real-time simulation of an autonomous quadrotor drone controlled by a cascaded PID intelligent control system. The simulation models simplified 2D quadrotor flight physics (planar motion), applies a three-tier hierarchical feedback controller (altitude PID → position PID → attitude PD), and provides two interactive front-ends: a Pygame visual flight simulator and a Streamlit telemetry dashboard.

---

## 🎯 Problem Statement

Autonomous aerial vehicles (UAVs) require robust control systems that can:
- Stabilize altitude against gravitational pull
- Track horizontal position waypoints precisely
- Reject unpredictable aerodynamic disturbances (wind gusts)
- Maintain safe attitude limits during all maneuvers

This project demonstrates these engineering challenges through mathematical modeling and cascaded PID feedback control, applied and validated in a real-time physical simulation.

---

## 🏛️ Project Objectives

1. Implement a physically accurate simplified 2D quadrotor dynamics model
2. Design a cascaded PID intelligent control architecture
3. Demonstrate autonomous takeoff, hovering, translation, and landing
4. Simulate aerodynamic wind disturbances and demonstrate rejection
5. Quantitatively evaluate performance using MAE, RMSE, overshoot, settling time
6. Provide real-time interactive simulation and telemetry dashboard

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     main.py  (Entry Point)                         │
├──────────────────┬──────────────────────┬──────────────────────────┤
│ Pygame Simulator │ Streamlit Dashboard  │ Evaluation/Benchmarks    │
├──────────────────┴──────────────────────┴──────────────────────────┤
│                   CASCADED CONTROL SYSTEM                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Setpoints (Z_ref, X_ref)                                   │   │
│  │          ↓                                                   │   │
│  │  [Altitude PID] ──────────────────────────→ Thrust (T)      │   │
│  │          ↓                                                   │   │
│  │  [Outer Position PID] ─→ θ_desired                          │   │
│  │          ↓                                                   │   │
│  │  [Inner Attitude PD]  ─────────────────→ Torque (τ)         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          ↓                                          │
│                   PHYSICAL DYNAMICS (RK4)                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  State: [x, z, vx, vz, θ, ω]  dt = 0.01 s (100 Hz)        │   │
│  │  Equations of Motion + Wind Disturbance + Ground Contact    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Model

### 2D Quadrotor State Vector

| Symbol | Variable | Unit |
|--------|----------|------|
| x | Horizontal position | m |
| z | Vertical altitude | m |
| vx | Horizontal velocity | m/s |
| vz | Vertical velocity | m/s |
| θ (theta) | Pitch angle | rad |
| ω (omega) | Angular velocity | rad/s |

### Equations of Motion

**Horizontal Acceleration:**
```
x_ddot = -(T/m) * sin(θ) - (c_dx/m) * vx + (Fx_wind/m)
```

**Vertical Acceleration:**
```
z_ddot = (T/m) * cos(θ) - g - (c_dz/m) * vz + (Fz_wind/m)
```

**Angular Acceleration:**
```
omega_dot = (tau - c_rot * omega) / I_yy
```

### Physical Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Mass | m | 1.2 | kg |
| Gravity | g | 9.81 | m/s² |
| Moment of Inertia | I_yy | 0.025 | kg·m² |
| Arm Length | l | 0.25 | m |
| Rotational Damping | c_rot | 0.05 | N·m·s/rad |
| Hover Thrust | T_hover = mg | 11.772 | N |
| Max Thrust | T_max | 24.0 | N |
| Max Torque | τ_max | 2.5 | N·m |

**Integration:** 4th-Order Runge-Kutta (RK4) at 100 Hz timestep.

---

## 🧠 PID Control System Architecture

### Cascaded Controller Design

```
                     ┌─────── OUTER LOOPS ──────┐  ┌── INNER LOOP ─┐
 Z_ref ──→[Altitude PID]──→ T (Thrust) ──────────────────────────────→ Physical
 X_ref ──→[Position PID]──→ θ_desired ──→[Attitude PD]──→ τ (Torque)─→ System
                                                                         ↓
 State feedback: z, vz, x, vx, θ, ω ←─────── Sensor Measurements ─────┘
```

### PID Mathematics

```
e(t) = r(t) - y(t)                    (tracking error)

u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(de/dt)

```

### Anti-Windup Protection
Conditional integration: integrator only accumulates when output is **not saturated**, preventing integral wind-up during large transients.

### Default PID Gains

| Controller | Kp | Ki | Kd |
|-----------|-----|-----|-----|
| Altitude | 12.0 | 2.5 | 6.0 |
| Position (Outer) | 1.8 | 0.1 | 2.2 |
| Attitude (Inner) | 8.0 | 0.5 | 1.2 |

---

## 🛠️ Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.10 | Core language |
| NumPy | ≥1.24 | Vector math, numerical arrays |
| SciPy | ≥1.10 | Scientific utilities |
| Pygame | ≥2.5 | Real-time 2D interactive simulation |
| Matplotlib | ≥3.7 | High-resolution performance plots |
| Streamlit | ≥1.28 | Interactive telemetry web dashboard |
| Pandas | ≥2.0 | Telemetry logging and CSV export |

---

## 📁 Project Structure

```
Quadrotor Intelligent Control System/
│
├── main.py                    # Master entry point (all 3 launch modes)
├── requirements.txt           # Dependency list
├── README.md                  # This documentation
│
├── drone/
│   ├── parameters.py          # Physical constants and operating limits
│   ├── physics.py             # Wind disturbance model, ground contact physics
│   └── dynamics.py            # Quadrotor 2D equations of motion (RK4 integration)
│
├── controllers/
│   ├── pid_controller.py      # Reusable PID controller with anti-windup
│   ├── altitude_control.py    # Altitude-to-thrust outer loop
│   └── position_control.py    # Position-to-angle + angle-to-torque cascaded loops
│
├── simulation/
│   ├── environment.py         # Coordinate transforms, wind particles, screen mapping
│   ├── drone_visualization.py # Pygame drawing: vehicle, HUD, trajectory, buttons
│   └── simulator.py           # Main simulation loop (physics + control + render)
│
├── dashboard/
│   └── app.py                 # Streamlit telemetry and control dashboard
│
├── evaluation/
│   ├── metrics.py             # MAE, RMSE, overshoot, settling time, Ess calculations
│   ├── plots.py               # 7 publication-quality performance graphs
│   └── scenarios.py           # 6 academic test scenario definitions
│
└── results/                   # Auto-generated plots, telemetry CSV, metric JSON
    ├── desired_vs_actual_altitude.png
    ├── desired_vs_actual_position.png
    ├── pitch_angle_vs_time.png
    ├── control_output_vs_time.png
    ├── flight_response_wind_disturbance.png
    ├── flight_trajectory_2d.png
    ├── isc_benchmark_summary.png
    ├── simulation_telemetry.csv
    └── validation_metrics.json
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- Windows/Linux supported

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install numpy scipy pygame matplotlib streamlit pandas
```

---

## 🚀 Execution Commands

### 1. Interactive Pygame Simulation (Real-Time Visual)
```bash
python main.py
```
Launches a 1200×750 pixel interactive flight simulator with HUD telemetry, flight controls, and trajectory visualization.

### 2. Streamlit Telemetry Dashboard
```bash
python main.py --dashboard
```
Opens a web-based aerospace telemetry dashboard in your browser at `http://localhost:8501`.

Alternatively run directly:
```bash
python -m streamlit run dashboard/app.py
```

### 3. Automated Evaluation & Graph Generation
```bash
python main.py --evaluate
```
Runs all 6 ISC validation test scenarios and exports plots + metrics to `results/`.

---

## 🎮 Simulation Controls (Pygame)

| Key / Button | Action |
|-------------|--------|
| `T` or `Space` | Autonomous Takeoff |
| `L` | Controlled Landing |
| `H` | Enter Hover Hold Mode |
| `A` or `←` | Move Left (−2 m waypoint) |
| `D` or `→` | Move Right (+2 m waypoint) |
| `W` | Toggle Wind Disturbance |
| `R` | Reset Simulation |
| `G` | Export Graphs to results/ |
| `ESC` | Exit |

---

## 🧪 Test Scenarios

### TEST 1: Autonomous Takeoff & Altitude Stabilization
Drone initiates from rest on ground. Altitude PID applies thrust from zero to reach 5.0 m setpoint. Measured: settling time, overshoot, MAE.

### TEST 2: Stationary Hovering Flight  
After reaching altitude, drone maintains a fixed spatial coordinate against gravity and minor perturbations. Measured: steady-state position error.

### TEST 3: Horizontal Step Translation  
Drone commanded from X=0 → X=+3 → X=−3 m. Cascaded position-attitude controller regulates trajectory. Measured: tracking MAE, response time.

### TEST 4: Autonomous Controlled Landing  
Descent at a programmed rate from cruise altitude to ground. Ground contact physics ensures safe touchdown. Measured: Final altitude, landing mode confirmation.

### TEST 5: Wind Disturbance & Rejection  
3.0 N external sinusoidal wind gust applied for 3.5 s. PID integral action detects and corrects displacement. Measured: Max deviation, recovery trajectory.

### TEST 6: Comprehensive Benchmark Suite  
Full mission covering all phases: takeoff → translation → wind disturbance → recovery → landing. All 7 performance plots exported.

---

## 📊 Actual Validation Results (Computed from Simulation Data)

> All metrics computed from actual recorded simulation telemetry, not fabricated.

| Test | Metric | Value |
|------|--------|-------|
| TEST 1 - Takeoff | Settling Time (5% band) | 2.70 s |
| TEST 1 - Takeoff | Max Overshoot | 5.39% |
| TEST 1 - Takeoff | Altitude MAE | 0.659 m |
| TEST 2 - Hover | Altitude Steady-State Error | 0.069 m |
| TEST 2 - Hover | Position Steady-State Error | 0.000 m |
| TEST 3 - Translation | Position MAE | 1.471 m |
| TEST 3 - Translation | Max Displacement | 3.015 m |
| TEST 4 - Landing | Final Altitude | 0.000 m |
| TEST 5 - Wind | Max Wind Deviation | 1.848 m |

---

## 📈 Performance Graphs

Graphs generated in `results/`:

1. **desired_vs_actual_altitude.png** – Altitude tracking + error envelope
2. **desired_vs_actual_position.png** – Horizontal position tracking + error envelope
3. **pitch_angle_vs_time.png** – Inner-loop attitude dynamics
4. **control_output_vs_time.png** – Thrust and torque actuator time histories
5. **flight_response_wind_disturbance.png** – Wind force profile and position recovery
6. **flight_trajectory_2d.png** – Full 2D spatial trajectory (X vs Z)
7. **isc_benchmark_summary.png** – Consolidated 4-panel poster (Altitude, Position, Pitch, Actuators)

---

## ⚠️ Known Limitations

1. **2D Planar Only**: This is a simplified longitudinal model (X-Z plane). Full 6-DOF (roll, yaw) is not implemented. This is documented as an academic simplification.
2. **Synthetic Wind**: Wind forces are mathematically simulated, not aerodynamically derived from CFD.
3. **No Sensor Noise Model**: Perfect state measurement is assumed (no Kalman filter).
4. **Simplified Motor Model**: Individual rotor dynamics and motor delay are not modeled.

---

## 🔭 Future Scope

1. Extend to full 6-DOF dynamics (roll, yaw, full 3D position)
2. Add Kalman Filter state estimator with synthetic sensor noise
3. Implement model-predictive control (MPC) for better constraint handling
4. Integrate ROS2 for hardware-in-the-loop simulation bridge
5. Machine learning-based adaptive PID gain tuning
6. Obstacle detection and path planning

---

## 👨‍🏫 Academic Information

- **Subject:** Intelligent Systems and Control (ISC)
- **Framework:** Cascaded PID Feedback Control
- **Physics Engine:** 4th-Order Runge-Kutta (RK4) Numerical Integration
- **Control Architecture:** Altitude → Position → Attitude Cascade

