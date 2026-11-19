"""
evaluate_data_driven.py
=======================
Evaluation & Benchmarking of MAPPO on the Data-Driven Neural World Model.
Zero SUMO / TraCI dependency — runs 100% on real-world Indian dataset surrogate.

Compares:
  1. MAPPO (Data-Driven Trained Policy from best_actor.pth)
  2. Fixed-Time (Cycle 30s green per phase)
  3. Random Controller (Baseline)

Across 6 Traffic Scenarios:
  low, normal, rush_hour_am, rush_hour_pm, holiday, incident

Outputs:
  - mappo_logs/data_driven_eval_summary.csv
  - mappo_models/data_driven_eval_comparison.png
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_driven_env import DataDrivenTrafficEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

MODEL_PATH = "mappo_models/best_actor.pth"
SCENARIOS = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]
N_EPISODES = 3
EPISODE_STEPS = 200
FIXED_PHASE_STEPS = 10  # 10 steps = 30 seconds per phase
PUNE_PCU_FACTOR = 0.850  # Average PCU weight from Pune Mendeley dataset


def load_actor():
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    if os.path.exists(MODEL_PATH):
        actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        print(f"[OK] Loaded trained MAPPO actor from {MODEL_PATH}")
    else:
        print(f"[!] Warning: {MODEL_PATH} not found. Using untrained actor.")
    actor.eval()
    return actor


def run_episode(env, controller_type, actor=None):
    obs_dict, _ = env.reset()
    total_reward = 0.0
    queues_pcu = []
    delays_sec = []
    throughputs = []

    fixed_phase = 0
    fixed_step_counter = 0

    for step in range(EPISODE_STEPS):
        actions = {}
        for i, tl in enumerate(TL_IDS):
            if controller_type == "mappo":
                obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act, _, _ = actor.get_action(obs_t, deterministic=True)
                actions[tl] = int(act.item())
            elif controller_type == "fixed":
                actions[tl] = fixed_phase
            else:  # random
                actions[tl] = np.random.randint(0, N_ACTIONS)

        if controller_type == "fixed":
            fixed_step_counter += 1
            if fixed_step_counter >= FIXED_PHASE_STEPS:
                fixed_phase = (fixed_phase + 1) % N_ACTIONS
                fixed_step_counter = 0

        obs_dict, state, reward, done, _ = env.step(actions)
        total_reward += reward

        # Extract corridor physical metrics from normalized state
        # 8 queue lanes * 40 max capacity * Pune modal split PCU factor
        tl_queues = []
        tl_delays = []
        tl_thrus = []
        for i in range(len(TL_IDS)):
            base = i * LOCAL_OBS_DIM
            q_norm = np.mean(state[base:base + 8])
            w_norm = np.mean(state[base + 8:base + 16])
            th_norm = np.sum(state[base + 16:base + 20])

            tl_queues.append(q_norm * 40.0 * PUNE_PCU_FACTOR)
            tl_delays.append(w_norm * 120.0)
            tl_thrus.append(th_norm * 450.0)

        queues_pcu.append(np.mean(tl_queues))
        delays_sec.append(np.mean(tl_delays))
        throughputs.append(np.mean(tl_thrus))

        if done:
            break

    return {
        "mean_queue_pcu": float(np.mean(queues_pcu)),
        "mean_delay_s": float(np.mean(delays_sec)),
        "mean_throughput_pcu_hr": float(np.mean(throughputs)),
        "episode_reward": float(total_reward),
        "mean_step_reward": float(total_reward / EPISODE_STEPS)
    }


def main():
    print("=" * 65)
    print("  MAPPO Data-Driven Evaluation (Simulator-Free Benchmark)")
    print("  Powered by Real-World Delhi Traffic Density + Pune Counts")
    print("=" * 65)

    actor = load_actor()
    env = DataDrivenTrafficEnv()

    controllers = ["mappo", "fixed", "random"]
    records = []

    for sc in SCENARIOS:
        print(f"\n--- Scenario: {sc.upper()} ---")
        for ctrl in controllers:
            ctrl_records = []
            for ep in range(N_EPISODES):
                env.scenario_name = sc
                res = run_episode(env, ctrl, actor)
                ctrl_records.append(res)

            mean_q = np.mean([r["mean_queue_pcu"] for r in ctrl_records])
            mean_d = np.mean([r["mean_delay_s"] for r in ctrl_records])
            mean_th = np.mean([r["mean_throughput_pcu_hr"] for r in ctrl_records])
            mean_rw = np.mean([r["mean_step_reward"] for r in ctrl_records])

            records.append({
                "scenario": sc,
                "controller": ctrl,
                "queue_pcu": round(mean_q, 2),
                "delay_sec": round(mean_d, 2),
                "throughput_pcu_hr": round(mean_th, 1),
                "step_reward": round(mean_rw, 3)
            })
            print(f"  {ctrl.upper():<7} | Queue: {mean_q:5.2f} PCU | Delay: {mean_d:5.2f}s | Thru: {mean_th:6.1f} PCU/h | Reward: {mean_rw:+6.3f}")

    df = pd.DataFrame(records)
    os.makedirs("mappo_logs", exist_ok=True)
    os.makedirs("mappo_models", exist_ok=True)
    csv_path = "mappo_logs/data_driven_eval_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved evaluation CSV to {csv_path}")

    # Generate Publication-Quality Visual Comparison Plot
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    scenarios = SCENARIOS
    x = np.arange(len(scenarios))
    w = 0.25

    colors = {
        "mappo": "#10b981",    # Emerald green
        "fixed": "#3b82f6",    # Blue
        "random": "#ef4444"    # Red
    }

    metrics = [
        ("queue_pcu", "Average Queue Length (PCU)", gs[0, 0], "PCU"),
        ("delay_sec", "Average Waiting Delay (seconds)", gs[0, 1], "Seconds"),
        ("throughput_pcu_hr", "Corridor Throughput (PCU / hr)", gs[1, 0], "PCU/hr"),
        ("step_reward", "Mean Step Reward (IRC Cost Formulation)", gs[1, 1], "Reward")
    ]

    for metric_col, title, grid_pos, unit in metrics:
        ax = fig.add_subplot(grid_pos)
        for i, ctrl in enumerate(controllers):
            subset = df[df["controller"] == ctrl]
            vals = [subset[subset["scenario"] == s][metric_col].values[0] for s in scenarios]
            ax.bar(x + (i - 1) * w, vals, width=w, label=ctrl.upper(), color=colors[ctrl], alpha=0.88, edgecolor="black", linewidth=0.6)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if grid_pos == gs[0, 0]:
            ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    plt.suptitle("Multi-Agent PPO vs Baselines on Neural World Model (Delhi & Pune Datasets)", fontsize=14, fontweight="bold", y=0.98)
    png_path = "mappo_models/data_driven_eval_comparison.png"
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved comparison chart to {png_path}")

    # Calculate overall improvement percentage of MAPPO vs Fixed
    mappo_delay = df[df["controller"] == "mappo"]["delay_sec"].mean()
    fixed_delay = df[df["controller"] == "fixed"]["delay_sec"].mean()
    delay_reduc = ((fixed_delay - mappo_delay) / fixed_delay) * 100.0

    mappo_queue = df[df["controller"] == "mappo"]["queue_pcu"].mean()
    fixed_queue = df[df["controller"] == "fixed"]["queue_pcu"].mean()
    queue_reduc = ((fixed_queue - mappo_queue) / fixed_queue) * 100.0

    print("=" * 65)
    print(f"  OVERALL PERFORMANCE SUMMARY:")
    print(f"  Average Queue : MAPPO {mappo_queue:.2f} PCU vs Fixed {fixed_queue:.2f} PCU ({queue_reduc:+.1f}% reduction)")
    print(f"  Average Delay : MAPPO {mappo_delay:.2f}s  vs Fixed {fixed_delay:.2f}s  ({delay_reduc:+.1f}% reduction)")
    print("=" * 65)


if __name__ == "__main__":
    main()
