"""
live_traffic_controller.py
==========================
Multi-Agent Live Traffic Signal Controller for Indian Corridors (NH-48).
Simulates and displays all 3 corridor traffic lights:
  - TL-1: West Junction (cluster_266437400_266437558)
  - TL-2: Middle Corridor (317145283)
  - TL-3: East Junction (cluster_4162190061_4162190645)

Features:
  - Natural 3-color signal cycling: 🟢 GREEN -> 🟡 YELLOW -> 🔴 RED -> 🟢 GREEN
  - Multi-Agent coordination with trained MAPPO Actor network
  - Live video camera stream analyzing Indian vehicles (2W, Auto, Car, Bus, Truck)

Usage:
  python src/live_traffic_controller.py
  python src/live_traffic_controller.py --source 0               # Webcam
  python src/live_traffic_controller.py --source vehicle_detection/traffic.mp4
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from mappo_networks import Actor
from data_driven_env import DataDrivenTrafficEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS, N_AGENTS

# Add vehicle_detection to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vehicle_detection")))
from live_vision_pipeline import LiveVisionPipeline, IRC_PCU_MAP

MODEL_PATH = "mappo_models/best_actor.pth"
PUNE_PCU = 0.850

TL_LABELS = [
    {"name": "TL-1 (West)", "desc": "Sec 29 Entry"},
    {"name": "TL-2 (Mid)", "desc": "IFFCO Core"},
    {"name": "TL-3 (East)", "desc": "Cyber City Exit"}
]

PHASE_LABELS = {
    0: "Phase 0 (NH-48 Main)",
    1: "Phase 1 (Cross Arterial)",
    2: "Phase 2 (Protected Turn)"
}


def draw_corridor_hud(frame, metrics, env, mode="MAPPO", fps=30.0, monitored_idx=0):
    """
    Renders the clean 3-Intersection Corridor Multi-Agent HUD onto the video frame:
      - Top Banner with Indian vehicle counts and PCU
      - 3 Photorealistic Traffic Light Signal Housings (TL-1, TL-2, TL-3)
      - Distinct glowing Red, Yellow, Green lamps
      - Live queue and wait telemetry
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 1. Top HUD Banner ───────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 80), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    mode_color = (0, 220, 80) if mode == "MAPPO" else (0, 140, 255)
    mode_text = f"[{mode} MULTI-AGENT AI]" if mode == "MAPPO" else "[FIXED-TIME TRADITIONAL]"
    cv2.putText(frame, mode_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, mode_color, 2, cv2.LINE_AA)

    cv2.putText(frame, "NH-48 ARTERIAL CORRIDOR (3 CONNECTED TRAFFIC SIGNALS)",
                (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)

    c = metrics.get("counts", {})
    count_str = f"2W: {c.get('two_wheeler', 0)}  |  Auto: {c.get('auto_rickshaw', 0)}  |  Car: {c.get('car', 0)}  |  Bus: {c.get('bus', 0)}  |  Truck: {c.get('truck', 0)}  |  Flow: {metrics.get('total_pcu', 0.0):.1f} PCU"
    cv2.putText(frame, count_str, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)

    sig_summary = "  ──►  ".join([f"{TL_LABELS[i]['name']}: {env.phase_state[TL_IDS[i]]}" for i in range(3)])
    cv2.putText(frame, f"Signals: {sig_summary}", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    # ── 2. Render 3 Traffic Lights (Top Right) ──────────────────────────────────
    box_w = 95
    box_h = 175
    gap = 18
    start_x = w - (3 * box_w + 2 * gap + 16)
    start_y = 92

    for i, tl_info in enumerate(TL_LABELS):
        tl = TL_IDS[i]
        cur_p = env.current_phase[tl]
        sig_state = env.phase_state[tl]

        bx = start_x + i * (box_w + gap)
        by = start_y

        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (25, 28, 35), -1)
        border_col = (255, 200, 0) if i == monitored_idx else (70, 75, 85)
        border_th = 2 if i == monitored_idx else 1
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), border_col, border_th)

        cv2.putText(frame, tl_info["name"][:9], (bx + 8, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        cx = bx + box_w // 2
        r_y = by + 26
        y_y = by + 70
        g_y = by + 114
        radius = 16

        c_r = (0, 0, 255) if sig_state == "RED" else (0, 0, 60)
        c_y = (0, 230, 255) if sig_state == "YELLOW" else (0, 60, 60)
        c_g = (0, 255, 60) if sig_state == "GREEN" else (0, 60, 0)

        for cy, color, is_active in [(r_y, c_r, sig_state == "RED"),
                                     (y_y, c_y, sig_state == "YELLOW"),
                                     (g_y, c_g, sig_state == "GREEN")]:
            cv2.circle(frame, (cx, cy), radius + 2, (15, 15, 18), -1)
            cv2.circle(frame, (cx, cy), radius, color, -1)
            if is_active:
                cv2.circle(frame, (cx - 4, cy - 4), radius // 3, (255, 255, 255), -1)

        badge_col = (0, 0, 255) if sig_state == "RED" else ((0, 220, 255) if sig_state == "YELLOW" else (0, 255, 60))
        cv2.putText(frame, f"P{cur_p}: {sig_state}", (bx + 8, by + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_col, 1, cv2.LINE_AA)

        base = i * LOCAL_OBS_DIM
        q_pcu = float(np.mean(env.state[base:base + 8]) * 40.0 * PUNE_PCU)
        cv2.putText(frame, f"{q_pcu:.1f} PCU", (bx + 18, by + 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1, cv2.LINE_AA)

        if i < 2:
            arrow_x1 = bx + box_w + 3
            arrow_x2 = arrow_x1 + gap - 6
            arrow_y = by + box_h // 2
            cv2.arrowedLine(frame, (arrow_x1, arrow_y), (arrow_x2, arrow_y), (0, 200, 255), 2, tipLength=0.4)

    # ── 3. Bottom Control Hints Bar ──────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 26), (w, h), (14, 16, 22), -1)
    hint_text = "[M] AI/Fixed Mode   |   [C] Cycle Camera   |   [0, 1, 2] Force Phase   |   [Q] Quit"
    cv2.putText(frame, hint_text, (16, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Live Traffic Signal Controller (NH-48 Corridor)")
    parser.add_argument("--source", type=str, default="vehicle_detection/intersection_video.mp4")
    parser.add_argument("--mode", type=str, default="mappo", choices=["mappo", "fixed"])
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 75)
    print("  MULTI-AGENT LIVE CORRIDOR TRAFFIC CONTROLLER (NH-48 ARTERIAL)")
    print("  3 Coordinated Signals: TL-1 (West) <---> TL-2 (Mid) <---> TL-3 (East)")
    print(f"  Input Source    : {args.source}")
    print(f"  Controller Mode : {args.mode.upper()}")
    print("=" * 75)

    vision = LiveVisionPipeline(conf_thresh=args.conf)
    env = DataDrivenTrafficEnv()
    obs_dict, state = env.reset()

    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    if os.path.exists(MODEL_PATH):
        actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        print(f"[OK] Loaded MAPPO Actor weights from {MODEL_PATH}")
    actor.eval()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture("vehicle_detection/synthetic_indian_traffic.mp4")

    mode = "MAPPO" if args.mode == "mappo" else "FIXED"
    step_timer = time.time()
    fixed_phase = 0
    fixed_counter = 0
    monitored_idx = 0
    prev_time = time.time()

    win_name = "Multi-Agent AI Traffic Corridor (NH-48: 3 Interconnected Signals)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1152, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            annotated_frame, metrics = vision.process_frame(frame)

            now = time.time()
            if (now - step_timer) >= 1.2:
                step_timer = now

                monitored_tl = TL_IDS[monitored_idx]
                live_q_norm = float(np.clip(metrics["queue_pcu"] / 35.0, 0.05, 0.95))
                base_idx = monitored_idx * LOCAL_OBS_DIM
                if env.current_phase[monitored_tl] == 0:
                    env.state[base_idx:base_idx + 4] = live_q_norm
                else:
                    env.state[base_idx + 4:base_idx + 8] = live_q_norm
                obs_dict = env._get_obs_dict()

                actions = {}
                for i, tl in enumerate(TL_IDS):
                    if mode == "MAPPO":
                        obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            act, _, _ = actor.get_action(obs_t, deterministic=True)
                        act_val = int(act.item())

                        base = i * LOCAL_OBS_DIM
                        main_q = np.mean(env.state[base:base + 4])
                        cross_q = np.mean(env.state[base + 4:base + 8])

                        if cross_q > main_q + 0.12 and env.steps_in_phase[tl] >= 4:
                            act_val = 1
                        elif main_q > cross_q + 0.12 and env.steps_in_phase[tl] >= 4:
                            act_val = 0
                        elif env.steps_in_phase[tl] >= 12:
                            act_val = (env.current_phase[tl] + 1) % 3

                        actions[tl] = act_val
                    else:
                        actions[tl] = fixed_phase

                if mode == "FIXED":
                    fixed_counter += 1
                    if fixed_counter >= 7:
                        fixed_phase = (fixed_phase + 1) % 3
                        fixed_counter = 0

                obs_dict, state, reward, done, info = env.step(actions)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            display_frame = draw_corridor_hud(annotated_frame, metrics, env, mode=mode, fps=fps, monitored_idx=monitored_idx)
            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('m') or key == ord('M'):
                mode = "FIXED" if mode == "MAPPO" else "MAPPO"
                print(f"[!] Controller mode toggled to: {mode}")
            elif key == ord('c') or key == ord('C'):
                monitored_idx = (monitored_idx + 1) % 3
                print(f"[!] Camera monitoring switched to: {TL_LABELS[monitored_idx]['name']}")
            elif key in [ord('0'), ord('1'), ord('2')]:
                forced_p = int(chr(key))
                for tl in TL_IDS:
                    env.current_phase[tl] = forced_p
                    env.phase_state[tl] = "GREEN"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Controller shut down cleanly.")


if __name__ == "__main__":
    main()
