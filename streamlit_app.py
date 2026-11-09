"""
╔══════════════════════════════════════════════════════════════════════╗
║  REAL-TIME SIMULATION OF INTELLIGENT CONTROL FOR A QUADROTOR DRONE  ║
║           Intelligent Systems & Control (ISC) Project                ║
║   Streamlit Cloud Deployment — Self-Contained, No pygame Required    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os, math, json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
#  §1  PHYSICAL PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DroneParams:
    mass:               float = 1.2
    arm_length:         float = 0.25
    I_yy:               float = 0.025
    gravity:            float = 9.81
    rotational_damping: float = 0.05
    linear_drag_x:      float = 0.15
    linear_drag_z:      float = 0.15
    thrust_min:         float = 0.0
    thrust_max:         float = 24.0
    torque_max:         float = 2.5
    max_tilt_rad:       float = np.radians(30.0)
    dt:                 float = 0.01

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity


# ═════════════════════════════════════════════════════════════════════════════
#  §2  PID CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════
class PIDController:
    def __init__(self, kp, ki, kd,
                 out_min=-np.inf, out_max=np.inf,
                 int_min=-np.inf, int_max=np.inf,
                 alpha=0.8):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.int_min, self.int_max = int_min, int_max
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.prev_deriv = 0.0
        self.has_run = False
        self.p = self.i = self.d = self.output = 0.0

    def compute(self, target, current, dt, rate=None):
        if dt <= 0:
            return 0.0
        err = target - current
        self.p = self.kp * err

        if rate is not None:
            deriv = -rate
        elif self.has_run:
            raw = (err - self.prev_err) / dt
            deriv = self.alpha * self.prev_deriv + (1 - self.alpha) * raw
        else:
            deriv = 0.0

        self.d = self.kd * deriv
        self.prev_deriv = deriv
        tentative = self.p + self.ki * self.integral + self.d
        sat_hi = tentative >= self.out_max and err > 0
        sat_lo = tentative <= self.out_min and err < 0
        if not (sat_hi or sat_lo):
            self.integral = float(np.clip(self.integral + err * dt,
                                          self.int_min, self.int_max))
        self.i = self.ki * self.integral
        self.output = float(np.clip(self.p + self.i + self.d,
                                    self.out_min, self.out_max))
        self.prev_err = err
        self.has_run = True
        return self.output


# ═════════════════════════════════════════════════════════════════════════════
#  §3  FLIGHT MODE STATE MACHINE
# ═════════════════════════════════════════════════════════════════════════════
class FlightMode(str, Enum):
    DISARMED  = "DISARMED"
    TAKEOFF   = "TAKEOFF"
    HOVER     = "HOVER"
    TRACKING  = "TRACKING"
    LANDING   = "LANDING"
    LANDED    = "LANDED"


# ═════════════════════════════════════════════════════════════════════════════
#  §4  WIND DISTURBANCE
# ═════════════════════════════════════════════════════════════════════════════
class WindDisturbance:
    def __init__(self):
        self.enabled      = False
        self.base_fx      = 0.0
        self.gust_amp     = 2.5
        self.gust_freq    = 0.5
        self.turb_std     = 0.3

    def compute(self, t: float) -> Tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        gust  = self.gust_amp * np.sin(2 * np.pi * self.gust_freq * t)
        noise = np.random.normal(0, self.turb_std)
        fx    = self.base_fx + gust + noise
        fz    = 0.15 * noise
        return float(fx), float(fz)


# ═════════════════════════════════════════════════════════════════════════════
#  §5  DYNAMICS (RK4)
# ═════════════════════════════════════════════════════════════════════════════
class QuadrotorDynamics:
    def __init__(self, p: DroneParams):
        self.p    = p
        self.wind = WindDisturbance()
        self.state = np.zeros(6)      # [x, z, vx, vz, theta, omega]
        self.thrust = self.torque = 0.0
        self.rotor_l = self.rotor_r  = 0.0
        self.time   = 0.0

    def reset(self, x=0.0, z=0.0):
        self.state[:] = [x, z, 0, 0, 0, 0]
        self.thrust = self.torque = self.time = 0.0
        self.rotor_l = self.rotor_r = 0.0

    # ── State accessors ──────────────────────────────────────────────────────
    @property
    def x(self):     return float(self.state[0])
    @property
    def z(self):     return float(self.state[1])
    @property
    def vx(self):    return float(self.state[2])
    @property
    def vz(self):    return float(self.state[3])
    @property
    def theta(self): return float(self.state[4])
    @property
    def theta_deg(self): return float(np.degrees(self.state[4]))
    @property
    def omega(self): return float(self.state[5])

    def _deriv(self, s, T, tau, fwx, fwz):
        _, z, vx, vz, th, om = s
        m, g   = self.p.mass, self.p.gravity
        I      = self.p.I_yy
        c_rot  = self.p.rotational_damping
        c_dx   = self.p.linear_drag_x
        c_dz   = self.p.linear_drag_z

        ax = -(T/m)*np.sin(th) - (c_dx/m)*vx + fwx/m
        az =  (T/m)*np.cos(th) - g - (c_dz/m)*vz + fwz/m
        if z <= 0 and az < 0:
            az = 0.0
        return np.array([vx, vz, ax, az, om, (tau - c_rot*om)/I])

    def step(self, thrust, torque, dt=None):
        dt = dt or self.p.dt
        T   = float(np.clip(thrust, self.p.thrust_min, self.p.thrust_max))
        tau = float(np.clip(torque, -self.p.torque_max, self.p.torque_max))
        self.thrust, self.torque = T, tau

        # Rotor forces
        l    = self.p.arm_length
        diff = tau / (2*l)
        self.rotor_l = max(0, T/2 - diff)
        self.rotor_r = max(0, T/2 + diff)

        fwx, fwz = self.wind.compute(self.time)

        # RK4
        k1 = self._deriv(self.state,            T, tau, fwx, fwz)
        k2 = self._deriv(self.state+0.5*dt*k1,  T, tau, fwx, fwz)
        k3 = self._deriv(self.state+0.5*dt*k2,  T, tau, fwx, fwz)
        k4 = self._deriv(self.state+    dt*k3,  T, tau, fwx, fwz)
        self.state += (dt/6)*(k1+2*k2+2*k3+k4)
        self.time  += dt

        # Ground contact
        if self.state[1] <= 0:
            self.state[1] = 0.0
            if self.state[3] < 0:
                self.state[3] = 0.0
            if T < 0.6 * self.p.hover_thrust:
                self.state[2] *= max(0, 1 - 15*dt)
                self.state[4] *= max(0, 1 - 20*dt)
                self.state[5]  = 0.0

        # Wrap angle
        self.state[4] = (self.state[4] + np.pi) % (2*np.pi) - np.pi
        return self.state.copy()


# ═════════════════════════════════════════════════════════════════════════════
#  §6  CASCADED FLIGHT CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════
class FlightController:
    def __init__(self, p: DroneParams,
                 alt_kp=12.0, alt_ki=2.5, alt_kd=6.0,
                 pos_kp=1.8,  pos_ki=0.1, pos_kd=2.2,
                 att_kp=8.0,  att_ki=0.5, att_kd=1.2):
        self.p = p
        self.alt = PIDController(alt_kp, alt_ki, alt_kd,
                                 out_min=-8, out_max=12,
                                 int_min=-3, int_max=3)
        self.pos = PIDController(pos_kp, pos_ki, pos_kd,
                                 out_min=-3.5, out_max=3.5,
                                 int_min=-1,   int_max=1)
        self.att = PIDController(att_kp, att_ki, att_kd,
                                 out_min=-p.torque_max, out_max=p.torque_max,
                                 int_min=-0.5, int_max=0.5)
        self.mode      = FlightMode.DISARMED
        self.target_x  = 0.0
        self.target_z  = 0.0
        self.descent_rate = 0.8    # m/s during landing
        self.desired_theta = 0.0

    def reset(self):
        self.alt.reset(); self.pos.reset(); self.att.reset()
        self.mode = FlightMode.DISARMED
        self.target_x = self.target_z = 0.0

    # ── Commands ─────────────────────────────────────────────────────────────
    def cmd_takeoff(self, z=5.0):
        self.target_z = max(1.0, z); self.mode = FlightMode.TAKEOFF

    def cmd_hover(self, x, z):
        self.target_x = x; self.target_z = max(0.5, z); self.mode = FlightMode.HOVER

    def cmd_move(self, dx):
        self.target_x += dx
        if self.mode not in (FlightMode.TAKEOFF, FlightMode.LANDING):
            self.mode = FlightMode.TRACKING

    def cmd_land(self):
        self.mode = FlightMode.LANDING

    def cmd_disarm(self):
        self.mode = FlightMode.DISARMED; self.target_z = 0.0

    # ── Control update ───────────────────────────────────────────────────────
    def update(self, x, z, vx, vz, theta, omega, dt) -> Tuple[float, float, float]:
        if self.mode == FlightMode.DISARMED:
            return 0.0, 0.0, 0.0
        if self.mode == FlightMode.LANDED:
            return (0.0, 0.0, 0.0) if z <= 0.05 else self._calc(x, z, vx, vz, theta, omega, dt)
        if self.mode == FlightMode.LANDING:
            self.target_z = max(0.0, self.target_z - self.descent_rate * dt)
            if z <= 0.08 and abs(vz) < 0.3:
                self.mode = FlightMode.LANDED
                return 0.0, 0.0, 0.0
        if self.mode == FlightMode.TAKEOFF:
            if abs(z - self.target_z) < 0.25 and abs(vz) < 0.2:
                self.mode = FlightMode.HOVER
        return self._calc(x, z, vx, vz, theta, omega, dt)

    def _calc(self, x, z, vx, vz, theta, omega, dt):
        # Altitude → thrust
        a_z = self.alt.compute(self.target_z, z, dt, rate=vz)
        cos_th = max(np.cos(theta), 0.3)
        T = float(np.clip(self.p.mass * (self.p.gravity + a_z) / cos_th,
                          self.p.thrust_min, self.p.thrust_max))

        # Position → desired pitch
        a_x = self.pos.compute(self.target_x, x, dt, rate=vx)
        T_eff = max(T, 0.4 * self.p.hover_thrust)
        sin_arg = float(np.clip(-(self.p.mass * a_x) / T_eff,
                                -np.sin(self.p.max_tilt_rad),
                                 np.sin(self.p.max_tilt_rad)))
        self.desired_theta = float(np.arcsin(sin_arg))

        # Attitude → torque
        tau = self.att.compute(self.desired_theta, theta, dt, rate=omega)
        return T, tau, self.desired_theta


# ═════════════════════════════════════════════════════════════════════════════
#  §7  SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class Simulator:
    def __init__(self, params: Optional[DroneParams] = None):
        self.p   = params or DroneParams()
        self.dyn = QuadrotorDynamics(self.p)
        self.ctl = FlightController(self.p)
        self.history: List[Dict] = []
        self.time = 0.0

    def reset(self):
        self.dyn.reset(); self.ctl.reset()
        self.history.clear(); self.time = 0.0

    def step(self, dt=None):
        dt = dt or self.p.dt
        T, tau, dth = self.ctl.update(
            self.dyn.x, self.dyn.z, self.dyn.vx, self.dyn.vz,
            self.dyn.theta, self.dyn.omega, dt)
        self.dyn.step(T, tau, dt)
        self.time += dt
        fwx, _ = self.dyn.wind.compute(self.time)
        rec = dict(
            time=round(self.time, 4),
            x=self.dyn.x, z=self.dyn.z,
            vx=self.dyn.vx, vz=self.dyn.vz,
            theta=self.dyn.theta, theta_deg=self.dyn.theta_deg,
            omega=self.dyn.omega,
            omega_deg=float(np.degrees(self.dyn.omega)),
            target_x=self.ctl.target_x, target_z=self.ctl.target_z,
            desired_theta_deg=float(np.degrees(dth)),
            thrust=self.dyn.thrust, thrust_pct=self.dyn.thrust/self.p.thrust_max*100,
            torque=self.dyn.torque,
            rotor_l=self.dyn.rotor_l, rotor_r=self.dyn.rotor_r,
            wind_fx=fwx, wind_enabled=self.dyn.wind.enabled,
            mode=self.ctl.mode.value,
        )
        self.history.append(rec)
        return rec

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def toggle_wind(self, base=2.0):
        self.dyn.wind.enabled = not self.dyn.wind.enabled
        if self.dyn.wind.enabled:
            self.dyn.wind.base_fx = base
        return self.dyn.wind.enabled


# ═════════════════════════════════════════════════════════════════════════════
#  §8  METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or len(df) < 10:
        return {}
    ez = df.target_z.values - df.z.values
    ex = df.target_x.values - df.x.values
    et = df.desired_theta_deg.values - df.theta_deg.values
    ss = max(5, int(len(df)*0.15))
    max_z = df.target_z.max()

    # Overshoot
    overshoot = max(0.0, (df.z.max()-max_z)/max_z*100) if max_z > 0.5 else 0.0

    # Settling (5% band of target)
    band = 0.05 * max_z if max_z > 0.5 else 0.05
    out_idx = np.where(np.abs(ez) > band)[0]
    if len(out_idx) == 0:
        settle = 0.0
    elif out_idx[-1] < len(df)-1:
        settle = float(df.time.iloc[out_idx[-1]+1])
    else:
        settle = float(df.time.iloc[-1])

    wind_rows = df[df.wind_enabled == True]
    wind_dev  = float(np.abs(wind_rows.x - wind_rows.target_x).max()) if not wind_rows.empty else 0.0

    return dict(
        alt_mae=round(float(np.mean(np.abs(ez))),4),
        alt_rmse=round(float(np.sqrt(np.mean(ez**2))),4),
        alt_overshoot=round(overshoot,2),
        alt_settling=round(settle,3),
        alt_ess=round(float(np.mean(np.abs(ez[-ss:]))),4),
        pos_mae=round(float(np.mean(np.abs(ex))),4),
        pos_rmse=round(float(np.sqrt(np.mean(ex**2))),4),
        pos_ess=round(float(np.mean(np.abs(ex[-ss:]))),4),
        att_mae=round(float(np.mean(np.abs(et))),4),
        att_rmse=round(float(np.sqrt(np.mean(et**2))),4),
        mean_thrust=round(float(df.thrust.mean()),3),
        max_thrust=round(float(df.thrust.max()),3),
        wind_dev=round(wind_dev,4),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  §9  MATPLOTLIB PLOTS (DARK AEROSPACE)
# ═════════════════════════════════════════════════════════════════════════════
_DARK = dict(
    figure_facecolor="#0F141C", axes_facecolor="#171F2C",
    axes_edgecolor="#2D3E54",   axes_labelcolor="#E0E6ED",
    xtick_color="#B0C0D0",      ytick_color="#B0C0D0",
    text_color="#FFFFFF",       grid_color="#233042",
)

def _apply():
    plt.rcParams.update({
        "figure.facecolor":_DARK["figure_facecolor"],
        "axes.facecolor":  _DARK["axes_facecolor"],
        "axes.edgecolor":  _DARK["axes_edgecolor"],
        "axes.labelcolor": _DARK["axes_labelcolor"],
        "axes.grid":True, "grid.color":"#233042",
        "grid.linestyle":"--","grid.alpha":0.5,
        "xtick.color":  _DARK["xtick_color"],
        "ytick.color":  _DARK["ytick_color"],
        "text.color":   _DARK["text_color"],
        "legend.facecolor":"#1E293B","legend.edgecolor":"#334155",
        "font.size":10,
    })

def plot_altitude(df):
    _apply(); fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5),sharex=True,
                                        gridspec_kw={"height_ratios":[2.5,1]})
    a1.plot(df.time,df.target_z,"r--",lw=2,label="Z_ref (Desired)")
    a1.plot(df.time,df.z,"#00D2FF",lw=2.2,label="Z (Actual)")
    a1.set_ylabel("Altitude [m]")
    a1.set_title("Altitude Stabilization — Cascaded PID",fontweight="bold",pad=10)
    a1.legend(loc="lower right")
    err=df.target_z-df.z
    a2.plot(df.time,err,"#FF7675",lw=1.5,label="Error e_z")
    a2.axhline(0,color="#636E72",ls=":",lw=1)
    a2.set_xlabel("Time [s]"); a2.set_ylabel("Error [m]"); a2.legend()
    plt.tight_layout(); return fig

def plot_position(df):
    _apply(); fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5),sharex=True,
                                        gridspec_kw={"height_ratios":[2.5,1]})
    a1.plot(df.time,df.target_x,"r--",lw=2,label="X_ref (Desired)")
    a1.plot(df.time,df.x,"#2ECC71",lw=2.2,label="X (Actual)")
    a1.set_ylabel("Position X [m]")
    a1.set_title("Horizontal Position Tracking",fontweight="bold",pad=10)
    a1.legend(loc="lower right")
    err=df.target_x-df.x
    a2.plot(df.time,err,"#F39C12",lw=1.5,label="Error e_x")
    a2.axhline(0,color="#636E72",ls=":",lw=1)
    a2.set_xlabel("Time [s]"); a2.set_ylabel("Error [m]"); a2.legend()
    plt.tight_layout(); return fig

def plot_pitch(df):
    _apply(); fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(df.time,df.desired_theta_deg,"r--",lw=1.8,label="θ_des")
    ax.plot(df.time,df.theta_deg,"#E056FD",lw=2.2,label="θ (Actual)")
    ax.axhline(30, color="#FF4757",ls=":",lw=1.2,label="Limit ±30°")
    ax.axhline(-30,color="#FF4757",ls=":",lw=1.2)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pitch [deg]")
    ax.set_title("Attitude (Pitch) Inner-Loop Response",fontweight="bold",pad=10)
    ax.legend(); plt.tight_layout(); return fig

def plot_control(df):
    _apply(); fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5),sharex=True)
    a1.plot(df.time,df.thrust,"#FF9F43",lw=2,label="Thrust T [N]")
    from drone.parameters import DroneParameters as _DP
    _p=DroneParams()
    a1.axhline(_p.hover_thrust,color="#74B9FF",ls="--",lw=1.5,label=f"Hover ({_p.hover_thrust:.2f} N)")
    a1.axhline(_p.thrust_max, color="#EE5253",ls=":",lw=1,label="Max Saturation")
    a1.set_ylabel("Thrust [N]")
    a1.set_title("Actuator Control Outputs",fontweight="bold",pad=10); a1.legend()
    a2.plot(df.time,df.torque,"#54A0FF",lw=1.8,label="Torque τ [N·m]")
    a2.axhline( 2.5,color="#EE5253",ls=":",lw=1)
    a2.axhline(-2.5,color="#EE5253",ls=":",lw=1,label="Limit ±2.5 N·m")
    a2.set_xlabel("Time [s]"); a2.set_ylabel("Torque [N·m]"); a2.legend()
    plt.tight_layout(); return fig

def plot_wind(df):
    _apply(); fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5),sharex=True)
    a1.plot(df.time,df.wind_fx,"#FF6B6B",lw=2,label="Wind Fx [N]")
    a1.set_ylabel("Wind Force [N]")
    a1.set_title("Disturbance Rejection — Wind Gust Response",fontweight="bold",pad=10); a1.legend()
    a2.plot(df.time,df.target_x,"r--",lw=1.8,label="X_ref",alpha=0.8)
    a2.plot(df.time,df.x,"#1DD1A1",lw=2.2,label="X (correcting)")
    a2.set_xlabel("Time [s]"); a2.set_ylabel("Position X [m]"); a2.legend()
    plt.tight_layout(); return fig

def plot_traj(df):
    _apply(); fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(df.x,df.z,"#3B82F6",lw=2.2,label="Flight Path")
    ax.scatter([df.x.iloc[0]],[df.z.iloc[0]],color="#10B981",s=90,zorder=5,label="Origin")
    ax.scatter([df.x.iloc[-1]],[df.z.iloc[-1]],color="#EF4444",s=90,zorder=5,label="End")
    tgt=df[["target_x","target_z"]].drop_duplicates()
    ax.scatter(tgt.target_x,tgt.target_z,color="#FDCB6E",marker="X",s=100,label="Waypoints")
    ax.axhline(0,color="#64748B",lw=2.5,label="Ground")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("2D Spatial Flight Trajectory",fontweight="bold",pad=10)
    ax.set_ylim(-0.5,max(8,df.z.max()+1.5)); ax.legend()
    plt.tight_layout(); return fig

def plot_summary(df):
    _apply(); fig,axs=plt.subplots(2,2,figsize=(14,8))
    axs[0,0].plot(df.time,df.target_z,"r--",label="Z_ref")
    axs[0,0].plot(df.time,df.z,"#00D2FF",label="Z"); axs[0,0].set_title("Altitude Tracking"); axs[0,0].legend()
    axs[0,1].plot(df.time,df.target_x,"r--",label="X_ref")
    axs[0,1].plot(df.time,df.x,"#2ECC71",label="X"); axs[0,1].set_title("Position Tracking"); axs[0,1].legend()
    axs[1,0].plot(df.time,df.desired_theta_deg,"r--",label="θ_des")
    axs[1,0].plot(df.time,df.theta_deg,"#E056FD",label="θ"); axs[1,0].set_title("Pitch Attitude"); axs[1,0].legend()
    axs[1,0].set_xlabel("Time [s]")
    axs[1,1].plot(df.time,df.thrust,"#F59E0B",label="Thrust [N]")
    ax2=axs[1,1].twinx()
    ax2.plot(df.time,df.torque,"#38BDF8",alpha=0.85,label="Torque [N·m]"); ax2.set_ylabel("Torque [N·m]",color="#38BDF8")
    axs[1,1].set_title("Actuation Commands"); axs[1,1].set_xlabel("Time [s]"); axs[1,1].legend(loc="upper left")
    plt.suptitle("ISC Quadrotor Intelligent Control — Benchmark Summary",
                 fontsize=14,fontweight="bold",y=0.99)
    plt.tight_layout(); return fig

def plot_drone_diagram(x, z, theta, target_x, target_z, history_x, history_z):
    """Schematic top-down & side view of drone position with trajectory."""
    _apply()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw ground
    ax.axhline(0, color="#64748B", lw=3, zorder=1, label="Ground (z=0)")

    # Trajectory trail
    if len(history_x) > 1:
        ax.plot(history_x, history_z, color="#3B82F6", lw=1.8,
                alpha=0.7, label="Flight Trajectory", zorder=2)

    # Target waypoint
    ax.scatter([target_x], [target_z], color="#46CC85", s=150, marker="X",
               zorder=5, label=f"Waypoint ({target_x:.1f} m, {target_z:.1f} m)")

    # Drone body
    ax.scatter([x], [z], color="#00D2FF", s=200, zorder=6, label=f"Drone (x={x:.2f} m, z={z:.2f} m)")

    # Rotors (schematic arms in 2D side view)
    arm = 0.5
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    lx = x - arm*cos_t;  lz = z + arm*sin_t
    rx = x + arm*cos_t;  rz = z - arm*sin_t
    ax.plot([lx, rx], [lz, rz], color="#94A3B8", lw=5, zorder=5)
    for mx, mz, col in [(lx, lz, "#FF6B6B"), (rx, rz, "#2ECC71")]:
        ax.scatter([mx], [mz], color=col, s=120, zorder=7)
        # Thrust arrow (upward from each rotor)
        thrust_len = 0.6
        nx, nz = -sin_t, cos_t
        ax.annotate("", xy=(mx+thrust_len*nx, mz+thrust_len*nz),
                    xytext=(mx, mz),
                    arrowprops=dict(arrowstyle="->", color="#FF9F43", lw=2))

    # Altitude indicator
    ax.plot([x, x], [0, z], color="#FDCB6E", ls="--", lw=1.2, alpha=0.6, label=f"Alt: {z:.2f} m")

    ax.set_xlim(min(-8, x-4), max(8, x+4))
    ax.set_ylim(-0.5, max(8, z+2))
    ax.set_xlabel("Horizontal Position X [m]")
    ax.set_ylabel("Altitude Z [m]")
    ax.set_title(f"Live Drone Position  |  Pitch θ = {math.degrees(theta):.1f}°", fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  §10  STREAMLIT PAGE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quadrotor Intelligent Control — ISC",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color:#0b0f19; color:#e2e8f0; }
.card {
    background:linear-gradient(135deg,#131c2e 0%,#172338 100%);
    border:1px solid #24344d; border-radius:10px; padding:16px 20px;
    box-shadow:0 4px 14px rgba(0,0,0,0.4); margin-bottom:12px;
}
.lbl { font-size:11px; text-transform:uppercase; letter-spacing:1px;
       color:#8da2c0; margin-bottom:4px; font-weight:600; }
.val { font-size:26px; font-weight:700; color:#fff;
       font-family:"Consolas","Courier New",monospace; }
.sub { font-size:12px; color:#00D2FF; margin-top:4px; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:12px; font-weight:700; letter-spacing:.5px; }
.HOVER    { background:rgba(46,204,113,.2);  color:#2ecc71; border:1px solid #2ecc71; }
.TAKEOFF  { background:rgba(52,152,219,.2);  color:#3498db; border:1px solid #3498db; }
.LANDING  { background:rgba(230,126,34,.2);  color:#e67e22; border:1px solid #e67e22; }
.TRACKING { background:rgba(241,196,15,.2);  color:#f1c40f; border:1px solid #f1c40f; }
.DISARMED,.LANDED { background:rgba(231,76,60,.2); color:#e74c3c; border:1px solid #e74c3c; }
h1,h2,h3 { color:#f8fafc; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "sim" not in st.session_state:
    sim = Simulator()
    sim.ctl.cmd_takeoff(5.0)
    sim.run(500)   # ~5 s of autonomous takeoff
    sim.ctl.cmd_hover(0.0, sim.dyn.z)
    sim.run(300)   # 3 s hover
    st.session_state.sim = sim
    st.session_state.hist_x = [r["x"] for r in sim.history]
    st.session_state.hist_z = [r["z"] for r in sim.history]

sim: Simulator = st.session_state.sim


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚁 Flight Avionics Console")
    st.caption("Intelligent Systems & Control (ISC) — Real-Time Dashboard")
    st.markdown("---")

    st.subheader("🎯 Target Waypoints")
    t_z = st.slider("Target Altitude Z [m]",  0.5, 10.0, float(sim.ctl.target_z or 5.0), 0.5)
    t_x = st.slider("Target Position  X [m]", -10.0, 10.0, float(sim.ctl.target_x or 0.0), 0.5)

    st.markdown("---")
    st.subheader("🌪️ Atmospheric Disturbance")
    wind_on  = st.checkbox("Enable Wind Disturbance", value=sim.dyn.wind.enabled)
    wind_fx  = st.slider("Base Wind Force [N]",  -5.0, 5.0, float(sim.dyn.wind.base_fx or 2.0), 0.5)
    gust_amp = st.slider("Gust Amplitude [N]",    0.0, 5.0, float(sim.dyn.wind.gust_amp), 0.5)
    sim.dyn.wind.enabled  = wind_on
    sim.dyn.wind.base_fx  = wind_fx
    sim.dyn.wind.gust_amp = gust_amp

    st.markdown("---")
    st.subheader("⚙️ PID Gain Tuner")
    with st.expander("Altitude PID"):
        sim.ctl.alt.kp = st.slider("Kp", 1.0, 30.0, float(sim.ctl.alt.kp), 0.5, key="alt_kp")
        sim.ctl.alt.ki = st.slider("Ki", 0.0, 10.0, float(sim.ctl.alt.ki), 0.1, key="alt_ki")
        sim.ctl.alt.kd = st.slider("Kd", 0.5, 20.0, float(sim.ctl.alt.kd), 0.5, key="alt_kd")
    with st.expander("Position (Outer) PID"):
        sim.ctl.pos.kp = st.slider("Kp", 0.2, 10.0, float(sim.ctl.pos.kp), 0.1, key="pos_kp")
        sim.ctl.pos.ki = st.slider("Ki", 0.0,  2.0, float(sim.ctl.pos.ki), 0.05, key="pos_ki")
        sim.ctl.pos.kd = st.slider("Kd", 0.2, 10.0, float(sim.ctl.pos.kd), 0.1, key="pos_kd")
    with st.expander("Attitude (Inner) PID"):
        sim.ctl.att.kp = st.slider("Kp", 1.0, 30.0, float(sim.ctl.att.kp), 0.5, key="att_kp")
        sim.ctl.att.ki = st.slider("Ki", 0.0,  5.0, float(sim.ctl.att.ki), 0.1, key="att_ki")
        sim.ctl.att.kd = st.slider("Kd", 0.2, 10.0, float(sim.ctl.att.kd), 0.1, key="att_kd")


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER + COMMAND BAR
# ═════════════════════════════════════════════════════════════════════════════
st.title("🚁 Quadrotor Intelligent Control — Real-Time Simulation")
st.markdown("**Autonomous Flight · Cascaded PID Feedback · Aerodynamic Disturbance Rejection**")

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)

def _run_and_rerun(fn, steps=150):
    fn(); sim.run(steps)
    # Update trajectory history
    st.session_state.hist_x.extend([r["x"] for r in sim.history[-steps:]])
    st.session_state.hist_z.extend([r["z"] for r in sim.history[-steps:]])
    # Trim to last 2000 pts
    st.session_state.hist_x = st.session_state.hist_x[-2000:]
    st.session_state.hist_z = st.session_state.hist_z[-2000:]
    st.rerun()

with c1:
    if st.button("🚀 Takeoff", use_container_width=True):
        _run_and_rerun(lambda: sim.ctl.cmd_takeoff(t_z), 400)
with c2:
    if st.button("⏸️ Hover", use_container_width=True):
        _run_and_rerun(lambda: sim.ctl.cmd_hover(sim.dyn.x, sim.dyn.z), 200)
with c3:
    if st.button("⬅️ Left", use_container_width=True):
        _run_and_rerun(lambda: sim.ctl.cmd_move(-2.0), 300)
with c4:
    if st.button("➡️ Right", use_container_width=True):
        _run_and_rerun(lambda: sim.ctl.cmd_move(2.0), 300)
with c5:
    if st.button("🛬 Land", use_container_width=True):
        _run_and_rerun(lambda: sim.ctl.cmd_land(), 500)
with c6:
    if st.button("💨 Wind Gust", use_container_width=True):
        sim.toggle_wind(wind_fx); sim.run(200)
        st.session_state.hist_x.extend([r["x"] for r in sim.history[-200:]])
        st.session_state.hist_z.extend([r["z"] for r in sim.history[-200:]])
        st.rerun()
with c7:
    if st.button("▶ Step +2s", use_container_width=True):
        sim.run(200)
        st.session_state.hist_x.extend([r["x"] for r in sim.history[-200:]])
        st.session_state.hist_z.extend([r["z"] for r in sim.history[-200:]])
        st.rerun()
with c8:
    if st.button("⚠️ Reset", use_container_width=True):
        new_sim = Simulator()
        st.session_state.sim = new_sim
        st.session_state.hist_x = []
        st.session_state.hist_z = []
        st.rerun()

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE TELEMETRY CARDS
# ═════════════════════════════════════════════════════════════════════════════
last = sim.history[-1] if sim.history else sim.step()
mode = last["mode"]

m1,m2,m3,m4,m5,m6 = st.columns(6)
with m1:
    st.markdown(f"""<div class="card"><div class="lbl">Flight Mode</div>
    <div style="margin-top:6px"><span class="badge {mode}">{mode}</span></div>
    <div class="sub">t = {last["time"]:.2f} s</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="card"><div class="lbl">Altitude Z</div>
    <div class="val">{last["z"]:.2f} <span style="font-size:16px">m</span></div>
    <div class="sub">Target: {last["target_z"]:.1f} m &nbsp;Δ {last["z"]-last["target_z"]:+.2f} m</div></div>""",
    unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="card"><div class="lbl">Position X</div>
    <div class="val">{last["x"]:.2f} <span style="font-size:16px">m</span></div>
    <div class="sub">Target: {last["target_x"]:.1f} m &nbsp;Δ {last["x"]-last["target_x"]:+.2f} m</div></div>""",
    unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="card"><div class="lbl">Vertical Vz</div>
    <div class="val">{last["vz"]:+.2f} <span style="font-size:16px">m/s</span></div>
    <div class="sub">Horiz Vx: {last["vx"]:+.2f} m/s</div></div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="card"><div class="lbl">Pitch Angle θ</div>
    <div class="val">{last["theta_deg"]:+.2f}°</div>
    <div class="sub">Desired: {last["desired_theta_deg"]:+.2f}° | ω: {last["omega_deg"]:+.1f}°/s</div></div>""",
    unsafe_allow_html=True)
with m6:
    st.markdown(f"""<div class="card"><div class="lbl">Total Thrust T</div>
    <div class="val">{last["thrust"]:.2f} <span style="font-size:16px">N</span></div>
    <div class="sub">{last["thrust_pct"]:.1f}% Max | τ: {last["torque"]:+.3f} N·m</div></div>""",
    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🛩️ Live Position View",
    "📈 Altitude & Position",
    "⚙️ Attitude & Control",
    "🌪️ Disturbance & 2D Path",
    "📊 Performance Metrics",
    "🧪 ISC Benchmarks",
])

df = sim.df()

with tab1:
    st.subheader("Real-Time Drone Position (Schematic)")
    hist_x = st.session_state.hist_x[-500:]
    hist_z = st.session_state.hist_z[-500:]
    fig_pos = plot_drone_diagram(sim.dyn.x, sim.dyn.z, sim.dyn.theta,
                                 sim.ctl.target_x, sim.ctl.target_z,
                                 hist_x, hist_z)
    st.pyplot(fig_pos, use_container_width=True)
    plt.close(fig_pos)

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Altitude", f"{sim.dyn.z:.3f} m", f"{sim.dyn.vz:+.2f} m/s")
    with info_col2:
        st.metric("Position X", f"{sim.dyn.x:.3f} m", f"{sim.dyn.vx:+.2f} m/s")
    with info_col3:
        st.metric("Pitch Angle", f"{sim.dyn.theta_deg:+.2f}°",
                  f"ω = {float(np.degrees(sim.dyn.omega)):+.1f}°/s")

with tab2:
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Altitude Tracking")
            st.pyplot(plot_altitude(df), use_container_width=True)
        with col_b:
            st.subheader("Horizontal Position Tracking")
            st.pyplot(plot_position(df), use_container_width=True)

with tab3:
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Pitch Attitude Response")
            st.pyplot(plot_pitch(df), use_container_width=True)
        with col_b:
            st.subheader("Actuator Commands (Thrust & Torque)")
            st.pyplot(plot_control(df), use_container_width=True)

with tab4:
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Wind Disturbance & Position Recovery")
            st.pyplot(plot_wind(df), use_container_width=True)
        with col_b:
            st.subheader("2D Spatial Trajectory (X vs Z)")
            st.pyplot(plot_traj(df), use_container_width=True)

with tab5:
    st.subheader("📊 Quantitative Control Performance Metrics")
    st.caption("All values computed from actual recorded simulation telemetry — no fabricated data.")
    if not df.empty:
        m = compute_metrics(df)
        ma, mb, mc = st.columns(3)
        with ma:
            st.markdown("### 🎯 Altitude Control")
            st.metric("MAE",            f"{m.get('alt_mae',0):.4f} m")
            st.metric("RMSE",           f"{m.get('alt_rmse',0):.4f} m")
            st.metric("Max Overshoot",  f"{m.get('alt_overshoot',0):.2f} %")
            st.metric("Settling Time",  f"{m.get('alt_settling',0):.3f} s")
            st.metric("Steady-State Error (Ess)", f"{m.get('alt_ess',0):.4f} m")
        with mb:
            st.markdown("### 📍 Horizontal Position")
            st.metric("MAE",            f"{m.get('pos_mae',0):.4f} m")
            st.metric("RMSE",           f"{m.get('pos_rmse',0):.4f} m")
            st.metric("Steady-State Error (Ess)", f"{m.get('pos_ess',0):.4f} m")
            if m.get('wind_dev',0) > 0:
                st.metric("Max Wind Deviation", f"{m.get('wind_dev',0):.4f} m")
        with mc:
            st.markdown("### ✈️ Attitude & Effort")
            st.metric("Pitch MAE",         f"{m.get('att_mae',0):.4f}°")
            st.metric("Pitch RMSE",        f"{m.get('att_rmse',0):.4f}°")
            st.metric("Mean Thrust",       f"{m.get('mean_thrust',0):.3f} N")
            st.metric("Peak Thrust",       f"{m.get('max_thrust',0):.3f} N")

        st.markdown("---")
        st.subheader("📈 Consolidated Performance Summary")
        st.pyplot(plot_summary(df), use_container_width=True)

        st.markdown("---")
        st.subheader("📥 Export Flight Data")
        csv = df.to_csv(index=False)
        st.download_button("⬇ Download Telemetry CSV",
                           data=csv, file_name="quadrotor_telemetry.csv",
                           mime="text/csv", use_container_width=True)
        js = json.dumps(m, indent=4)
        st.download_button("⬇ Download Metrics JSON",
                           data=js, file_name="flight_metrics.json",
                           mime="application/json", use_container_width=True)

with tab6:
    st.subheader("🧪 ISC Academic Validation Scenarios")
    st.markdown("""
Run any of the 6 standard ISC control system test scenarios.
Each scenario simulates a full flight experiment and displays results on the dashboard.
    """)
    scenario = st.selectbox("Select Validation Scenario", [
        "TEST 1 — Autonomous Takeoff & Altitude Stabilization",
        "TEST 2 — Stationary Hovering Flight",
        "TEST 3 — Horizontal Step Translation (Left & Right)",
        "TEST 4 — Autonomous Controlled Landing",
        "TEST 5 — Wind Disturbance & Automatic Recovery",
        "TEST 6 — Comprehensive Full-Mission Benchmark",
    ])

    if st.button("▶️ Run Scenario", use_container_width=True, type="primary"):
        with st.spinner("Simulating scenario (computing physics at 100 Hz)…"):
            new = Simulator()
            if "TEST 1" in scenario:
                new.ctl.cmd_takeoff(5.0); new.run(700)
            elif "TEST 2" in scenario:
                new.ctl.cmd_takeoff(4.0); new.run(400)
                new.ctl.cmd_hover(0.0,4.0); new.run(600)
            elif "TEST 3" in scenario:
                new.ctl.cmd_takeoff(4.0); new.run(300)
                new.ctl.cmd_move(3.0);  new.run(400)
                new.ctl.cmd_move(-6.0); new.run(500)
                new.ctl.cmd_move(3.0);  new.run(200)
            elif "TEST 4" in scenario:
                new.ctl.cmd_takeoff(4.0); new.run(400)
                new.ctl.cmd_land(); new.run(700)
            elif "TEST 5" in scenario:
                new.ctl.cmd_takeoff(5.0); new.run(300)
                new.toggle_wind(3.0); new.run(400)
                new.toggle_wind();    new.run(300)
            else:  # TEST 6 comprehensive
                new.ctl.cmd_takeoff(5.0); new.run(400)
                new.ctl.cmd_move(3.0); new.run(350)
                new.toggle_wind(2.5);  new.run(350)
                new.toggle_wind();     new.run(200)
                new.ctl.cmd_land();    new.run(600)

            st.session_state.sim = new
            st.session_state.hist_x = [r["x"] for r in new.history]
            st.session_state.hist_z = [r["z"] for r in new.history]
        st.success("✅ Scenario complete! Navigate to other tabs to view results.")
        st.rerun()

    st.markdown("---")
    st.markdown("""
| Test | Scenario | Key Metric |
|------|----------|------------|
| TEST 1 | Takeoff & Altitude Stabilization | Settling Time, Overshoot |
| TEST 2 | Stationary Hovering Flight | Steady-State Error Ess |
| TEST 3 | Horizontal Translation | Position MAE, Response Time |
| TEST 4 | Controlled Landing | Final Altitude = 0 m |
| TEST 5 | Wind Disturbance Rejection | Max Deviation, Recovery Time |
| TEST 6 | Comprehensive Benchmark | All Metrics |
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:12px'>"
    "🚁 Quadrotor ISC Project &nbsp;|&nbsp; Cascaded PID Control &nbsp;|&nbsp; "
    "RK4 Physics at 100 Hz &nbsp;|&nbsp; Streamlit Cloud Deployment"
    "</div>",
    unsafe_allow_html=True
)

