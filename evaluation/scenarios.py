"""
Automated Test Scenarios and Benchmark Suite.
Intelligent Systems and Control (ISC) Project.

Defines and executes the 6 academic validation test scenarios:
TEST 1: Autonomous Takeoff and Altitude Stabilization
TEST 2: Stationary Hovering Flight
TEST 3: Horizontal Translation (Left & Right Navigation)
TEST 4: Autonomous Controlled Landing
TEST 5: Aerodynamic Wind Disturbance & Recovery
TEST 6: Comprehensive Benchmark & Controller Performance Evaluation
"""

import os
import json
from typing import Dict, Any
import pandas as pd
from simulation.simulator import QuadrotorSimulator
from evaluation.metrics import calculate_flight_metrics
from evaluation.plots import generate_all_plots


def run_test_1_takeoff(duration_s: float = 7.0) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """TEST 1: Autonomous takeoff and altitude stabilization."""
    sim = QuadrotorSimulator()
    sim.controller.command_takeoff(target_altitude=5.0)

    steps = int(duration_s / sim.params.dt)
    for _ in range(steps):
        sim.step()

    df = sim.get_history_df()
    metrics = calculate_flight_metrics(df)
    return df, metrics


def run_test_2_hover(duration_s: float = 8.0) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """TEST 2: Maintaining a constant hovering position."""
    sim = QuadrotorSimulator()
    sim.controller.command_takeoff(target_altitude=4.0)

    # 3.5s takeoff
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    # Enter hover hold
    sim.controller.command_hover(current_x=0.0, current_z=4.0)
    for _ in range(int((duration_s - 3.5) / sim.params.dt)):
        sim.step()

    df = sim.get_history_df()
    metrics = calculate_flight_metrics(df)
    return df, metrics


def run_test_3_translation(duration_s: float = 12.0) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """TEST 3: Forward, backward, left, and right movement."""
    sim = QuadrotorSimulator()
    sim.controller.command_takeoff(target_altitude=4.0)
    for _ in range(int(3.0 / sim.params.dt)):
        sim.step()

    # Move right (+3m)
    sim.controller.command_move_relative(3.0)
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    # Move left (-6m -> setpoint -3m)
    sim.controller.command_move_relative(-6.0)
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    # Return to origin (0m)
    sim.controller.command_move_relative(3.0)
    for _ in range(int(2.0 / sim.params.dt)):
        sim.step()

    df = sim.get_history_df()
    metrics = calculate_flight_metrics(df)
    return df, metrics


def run_test_4_landing(duration_s: float = 10.0) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """TEST 4: Automatic landing."""
    sim = QuadrotorSimulator()
    sim.controller.command_takeoff(target_altitude=4.0)
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    # Initiate landing sequence
    sim.controller.command_land()
    for _ in range(int((duration_s - 3.5) / sim.params.dt)):
        sim.step()

    df = sim.get_history_df()
    metrics = calculate_flight_metrics(df)
    return df, metrics


def run_test_5_wind_disturbance(duration_s: float = 10.0) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """TEST 5: Wind disturbance and stabilization recovery."""
    sim = QuadrotorSimulator()
    sim.controller.command_takeoff(target_altitude=5.0)
    for _ in range(int(3.0 / sim.params.dt)):
        sim.step()

    # Turn on 3.0 N wind gust
    sim.toggle_wind(base_force=3.0)
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    # Turn off wind to test recovery back to target
    sim.toggle_wind()
    for _ in range(int(3.5 / sim.params.dt)):
        sim.step()

    df = sim.get_history_df()
    metrics = calculate_flight_metrics(df)
    return df, metrics


def run_all_validation_tests(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes all 6 validation scenarios, logs authentic metrics,
    and produces high-resolution plots in results/.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*70)
    print("  INTELLIGENT SYSTEMS & CONTROL (ISC) - VALIDATION BENCHMARKS")
    print("="*70)

    results_summary = {}

    print("\n[TEST 1/6] Running Autonomous Takeoff & Altitude Stabilization...")
    df1, m1 = run_test_1_takeoff()
    results_summary["TEST_1_Takeoff"] = m1
    print(f"  -> Alt Settling Time: {m1['altitude']['settling_time_s']} s | Overshoot: {m1['altitude']['max_overshoot_pct']}% | MAE: {m1['altitude']['MAE_m']} m")

    print("\n[TEST 2/6] Running Stationary Hovering Flight...")
    df2, m2 = run_test_2_hover()
    results_summary["TEST_2_Hover"] = m2
    print(f"  -> Steady-State Alt Error: {m2['altitude']['steady_state_error_m']} m | Pos Ess: {m2['position']['steady_state_error_m']} m")

    print("\n[TEST 3/6] Running Directional Step Translation (Left & Right)...")
    df3, m3 = run_test_3_translation()
    results_summary["TEST_3_Translation"] = m3
    print(f"  -> Pos MAE: {m3['position']['MAE_m']} m | Max Displacement: {m3['position']['max_x_displacement_m']} m")

    print("\n[TEST 4/6] Running Autonomous Controlled Landing...")
    df4, m4 = run_test_4_landing()
    results_summary["TEST_4_Landing"] = m4
    print(f"  -> Final Altitude: {df4['z'].iloc[-1]:.3f} m | Final Mode: {df4['mode'].iloc[-1]}")

    print("\n[TEST 5/6] Running Aerodynamic Wind Disturbance Rejection...")
    df5, m5 = run_test_5_wind_disturbance()
    results_summary["TEST_5_Disturbance"] = m5
    print(f"  -> Max Wind Deviation: {m5.get('disturbance', {}).get('max_wind_deviation_m', 'N/A')} m")

    print("\n[TEST 6/6] Running Comprehensive Integrated Scenario & Plot Generation...")
    # Composite mission covering all flight regimes for master report figures
    comp_sim = QuadrotorSimulator()
    comp_sim.controller.command_takeoff(5.0)
    for _ in range(350): comp_sim.step()
    comp_sim.controller.command_move_relative(3.0)
    for _ in range(300): comp_sim.step()
    comp_sim.toggle_wind(2.5)
    for _ in range(300): comp_sim.step()
    comp_sim.toggle_wind()
    for _ in range(200): comp_sim.step()
    comp_sim.controller.command_land()
    for _ in range(500): comp_sim.step()

    comp_df = comp_sim.get_history_df()
    comp_metrics = calculate_flight_metrics(comp_df)
    results_summary["TEST_6_Comprehensive"] = comp_metrics

    # Save all required graphs
    generate_all_plots(comp_df, output_dir=output_dir)

    # Save raw telemetry CSV and JSON metrics
    comp_df.to_csv(os.path.join(output_dir, "simulation_telemetry.csv"), index=False)
    with open(os.path.join(output_dir, "validation_metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n" + "="*70)
    print(f"  ALL BENCHMARK TESTS COMPLETE. Artifacts saved in '{output_dir}/'")
    print("="*70 + "\n")

    return results_summary

