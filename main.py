"""
Master Entry Point: Real-Time Simulation of Intelligent Control for a Quadrotor Drone.
Intelligent Systems and Control (ISC) Project.

Usage:
    python main.py             # Launches interactive Pygame simulation
    python main.py --dashboard # Launches interactive Streamlit telemetry dashboard
    python main.py --evaluate  # Executes all 6 academic benchmark tests and generates plots
"""

import sys
import os
import argparse
import subprocess

# Ensure current project directory is in python search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def launch_pygame_simulation() -> None:
    """Starts the real-time interactive Pygame visualizer and flight physics loop."""
    from simulation.simulator import QuadrotorSimulator
    print("[Main] Initializing Quadrotor Physics and Pygame Visualizer...")
    simulator = QuadrotorSimulator()
    simulator.run_interactive()


def launch_streamlit_dashboard() -> None:
    """Launches the Streamlit telemetry and parameter tuning web dashboard."""
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    print(f"[Main] Launching Streamlit Telemetry Dashboard from: {dashboard_path}")
    # Use current python executable to run streamlit module
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Main] Dashboard stopped by user.")


def run_evaluation_suite() -> None:
    """Executes the 6 academic validation test scenarios and generates plots."""
    from evaluation.scenarios import run_all_validation_tests
    output_dir = os.path.join(PROJECT_ROOT, "results")
    metrics = run_all_validation_tests(output_dir=output_dir)
    print(f"[Main] Performance evaluation successfully generated in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Simulation of Intelligent Control for a Quadrotor Drone (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Interactive 2D Pygame simulation with flight HUD
    python main.py --dashboard  # Streamlit aerospace telemetry dashboard
    python main.py --evaluate   # Run ISC benchmark tests and export report plots
        """
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the interactive Streamlit telemetry dashboard"
    )
    parser.add_argument(
        "--evaluate",
        "--test",
        dest="evaluate",
        action="store_true",
        help="Run the 6 automated ISC test scenarios and generate evaluation graphs"
    )

    args = parser.parse_args()

    if args.dashboard:
        launch_streamlit_dashboard()
    elif args.evaluate:
        run_evaluation_suite()
    else:
        launch_pygame_simulation()


if __name__ == "__main__":
    main()

