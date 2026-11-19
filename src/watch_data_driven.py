"""
watch_data_driven.py
====================
Real-Time Terminal Monitor & Playback for Data-Driven Multi-Agent Traffic Control.
Runs 100% Simulator-Free on the Neural World Model (Trained on Delhi & Pune Traffic Data).

Usage:
  python src/watch_data_driven.py
  python src/watch_data_driven.py --scenario rush_hour_am --steps 100 --delay 0.1
"""

import sys
import time
import argparse
import numpy as np
import torch

from data_driven_env import DataDrivenTrafficEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

MODEL_PATH = "mappo_models/best_actor.pth"
PUNE_PCU = 0.850

PHASE_NAMES = {
    0: "Phase 0 (North-South Through)",
    1: "Phase 1 (East-West Arterial)",
    2: "Phase 2 (Protected Turn / Diagonal)"
}


def make_bar(val, max_val=25.0, length=12):
    frac = min(max(val / max_val, 0.0), 1.0)
    filled = int(round(frac * length))
    return "#" * filled + "-" * (length - filled)


def main():
    parser = argparse.ArgumentParser(description="Watch MAPPO control the corridor on Neural World Model")
    parser.add_argument("--scenario", type=str, default="rush_hour_am", help="Traffic scenario")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps (0 for infinite)")
    parser.add_argument("--delay", type=float, default=0.08, help="Pacing delay per step in seconds")
    parser.add_argument("--controller", type=str, default="mappo", choices=["mappo", "fixed"], help="Controller policy")
    args = parser.parse_args()

    # Load Actor
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    try:
        actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    except Exception as e:
        print(f"[!] Warning: Could not load {MODEL_PATH} ({e}). Using initialized weights.")
    actor.eval()

    env = DataDrivenTrafficEnv(scenario_name=args.scenario)
    obs_dict, state = env.reset()

    print("\n" + "=" * 80)
    print(f"  DATA-DRIVEN NEURAL WORLD MODEL - LIVE CORRIDOR PLAYBACK")
    print(f"  Dataset Sources: IIT Delhi Traffic Density + Mendeley Pune Counts (PCU=0.850)")
    print(f"  Corridor: NH-48 Tri-Intersection Arterial (West - Middle - East)")
    print(f"  Scenario: {args.scenario} | Controller: {args.controller.upper()}")
    print("=" * 80 + "\n")
    time.sleep(1.0)

    step = 0
    fixed_phase = 0
    fixed_counter = 0

    try:
        while True:
            step += 1
            actions = {}
            for i, tl in enumerate(TL_IDS):
                if args.controller == "mappo":
                    obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        act, _, _ = actor.get_action(obs_t, deterministic=True)
                    actions[tl] = int(act.item())
                else:
                    actions[tl] = fixed_phase

            if args.controller == "fixed":
                fixed_counter += 1
                if fixed_counter >= 10:
                    fixed_phase = (fixed_phase + 1) % N_ACTIONS
                    fixed_counter = 0

            obs_dict, state, reward, done, info = env.step(actions)

            # Format per-intersection display
            # Clear / header print
            sim_time_sec = step * 3
            print(f"\n--- [Step {step:4d} | Sim Time: {sim_time_sec:4d}s | Scenario: {args.scenario}] " + "-" * 30)

            total_q = 0.0
            total_w = 0.0
            total_th = 0.0

            tl_labels = ["TL-1 (West Jct)", "TL-2 (Mid NH48)", "TL-3 (East Jct)"]

            for i, tl in enumerate(TL_IDS):
                base = i * LOCAL_OBS_DIM
                p_idx = int(round(state[base + 20] * 2.0))
                sig_state = "GREEN" if state[base + 21] >= 0.9 else "YELLOW"
                sig_badge = "[G]" if sig_state == "GREEN" else "[Y]"

                q_val = float(np.mean(state[base:base + 8]) * 40.0 * PUNE_PCU)
                w_val = float(np.mean(state[base + 8:base + 16]) * 120.0)
                th_val = float(np.sum(state[base + 16:base + 20]) * 450.0)

                total_q += q_val
                total_w += w_val
                total_th += th_val

                bar = make_bar(q_val, max_val=25.0, length=10)
                phase_str = PHASE_NAMES.get(p_idx, f"Phase {p_idx}")

                print(f"  {tl_labels[i]} {sig_badge} {sig_state:<6} | {phase_str:<32} | Queue: {bar} {q_val:5.1f} PCU | Wait: {w_val:4.1f}s")

            mean_q = total_q / 3.0
            mean_w = total_w / 3.0
            print(f"  >> Corridor Mean Queue: {mean_q:5.1f} PCU | Avg Delay: {mean_w:4.1f}s | Throughput: {total_th:6.1f} PCU/hr | Step Reward: {reward:+6.2f}")

            time.sleep(args.delay)

            if args.steps > 0 and step >= args.steps:
                break
            if done:
                obs_dict, state = env.reset()

        print("\n" + "=" * 80)
        print("  Playback completed successfully!")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")


if __name__ == "__main__":
    main()
