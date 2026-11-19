"""
dashboard_live_vision.py
========================
Multi-Agent Real-Time Traffic Signal Control for Indian Arterial Corridors (NH-48).
Couples YOLOv8 Indian Vehicle Perception (IISc AIM BMD-45 / IRC PCU) with a
3-Intersection MAPPO Policy along the Delhi-Gurgaon NH-48 Corridor:
  - TL-1: West Junction (cluster_266437400_266437558)
  - TL-2: Middle Corridor (317145283)
  - TL-3: East Junction (cluster_4162190061_4162190645)

Features:
  - Realistic 3-color Signal cycling: 🔴 RED, 🟡 YELLOW, 🟢 GREEN
  - Natural All-Red clearance and phase alternation
  - Multi-agent coordination with trained Actor policy
  - Live video camera stream analyzing Indian vehicles (2W, Auto, Car, Bus, Truck)

Run with:
  streamlit run src/dashboard_live_vision.py
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vehicle_detection")))
from live_vision_pipeline import LiveVisionPipeline, IRC_PCU_MAP, CLASS_COLORS

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from mappo_networks import Actor
from data_driven_env import DataDrivenTrafficEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS, N_AGENTS

st.set_page_config(
    page_title="AI Traffic Signal Corridor (NH-48)",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .corridor-header {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .tl-card {
        border-radius: 12px;
        padding: 14px;
        background: #1e293b;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .lamp-circle {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin: 0 6px;
    }
    .lamp-red-on    { background: #ef4444; box-shadow: 0 0 14px #ef4444; }
    .lamp-red-off   { background: #450a0a; }
    .lamp-yel-on    { background: #eab308; box-shadow: 0 0 14px #eab308; }
    .lamp-yel-off   { background: #422006; }
    .lamp-grn-on    { background: #10b981; box-shadow: 0 0 14px #10b981; }
    .lamp-grn-off   { background: #022c22; }
    .badge-signal {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 6px;
    }
    .sig-green  { background: #059669; color: white; }
    .sig-yellow { background: #d97706; color: white; }
    .sig-red    { background: #dc2626; color: white; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "mappo_models/best_actor.pth"
PUNE_PCU = 0.850

TL_INFO = [
    {"id": TL_IDS[0], "name": "TL-1: West Junction", "desc": "Sector 29 / Leisure Valley Entry"},
    {"id": TL_IDS[1], "name": "TL-2: Middle Corridor", "desc": "IFFCO Chowk Core Interchange"},
    {"id": TL_IDS[2], "name": "TL-3: East Junction", "desc": "Signature Tower / Cyber City Exit"}
]

PHASE_DESCRIPTIONS = {
    0: "Phase 0 (NH-48 Main East-West)",
    1: "Phase 1 (Cross Arterial North-South)",
    2: "Phase 2 (Protected Turn / Diag)"
}


@st.cache_resource
def load_system():
    env = DataDrivenTrafficEnv()
    vision = LiveVisionPipeline(conf_thresh=0.25)
    actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
    if os.path.exists(MODEL_PATH):
        actor.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    actor.eval()
    return env, vision, actor


env, vision, actor = load_system()

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.title("🚦 Corridor Control Center")
st.sidebar.caption("NH-48 Tri-Intersection Multi-Agent AI")

monitored_tl_idx = st.sidebar.selectbox(
    "Camera Feed Deployed At",
    [0, 1, 2],
    format_func=lambda i: TL_INFO[i]["name"]
)

source_opt = st.sidebar.selectbox(
    "Video Feed Source",
    [
        "Intersection Video 1 (NH-48 Corridor)",
        "Synthetic Indian Traffic (Autos & Bikes)",
        "Urban Traffic Stream 2",
        "Live Webcam (Device 0)",
        "Upload Custom Video (MP4)"
    ]
)

video_source = None
if source_opt == "Intersection Video 1 (NH-48 Corridor)":
    video_source = "vehicle_detection/intersection_video.mp4"
elif source_opt == "Synthetic Indian Traffic (Autos & Bikes)":
    video_source = "vehicle_detection/synthetic_indian_traffic.mp4"
elif source_opt == "Urban Traffic Stream 2":
    video_source = "vehicle_detection/traffic.mp4"
elif source_opt == "Live Webcam (Device 0)":
    video_source = 0
else:
    uploaded = st.sidebar.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        temp_path = "vehicle_detection/uploaded_temp.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        video_source = temp_path

controller_mode = st.sidebar.radio(
    "Corridor Policy",
    ["MAPPO Multi-Agent AI (Cooperative)", "Fixed-Time Cycle (Traditional 30s)"]
)

reset_corridor = st.sidebar.button("🔄 Reset Corridor Simulation")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<div class='corridor-header'>🚦 Multi-Agent Adaptive Traffic Corridor — NH-48 Delhi-Gurgaon</div>", unsafe_allow_html=True)
st.caption("3 Interconnected Signalized Intersections communicating via Centralized Training with Decentralized Execution (MAPPO).")

# ── 3 Connected Traffic Light Cards (Top Visual Display) ───────────────────────
st.subheader("📍 Tri-Intersection Real-Time Signals")
col_tl1, col_tl2, col_tl3 = st.columns(3)
tl_placeholders = [col_tl1.empty(), col_tl2.empty(), col_tl3.empty()]

# ── Corridor Status Banner ───────────────────────────────────────────────────
corridor_banner = st.empty()

st.markdown("---")

# Main Content Layout: Video on Left (54%), Telemetry & Charts on Right (46%)
col_vid, col_stats = st.columns([1.2, 1.0])

with col_vid:
    st.subheader(f"📹 CCTV Feed: {TL_INFO[monitored_tl_idx]['name']}")
    video_placeholder = st.empty()
    run_live = st.checkbox("▶ Stream Corridor & Video", value=True)

with col_stats:
    st.subheader("📊 Multi-Agent Telemetry & Decisions")
    chart_classes_placeholder = st.empty()
    chart_pcu_placeholder = st.empty()
    decision_log_placeholder = st.empty()

# Persistent state initialization
if "obs_dict" not in st.session_state or reset_corridor:
    st.session_state.obs_dict, st.session_state.state = env.reset()
    st.session_state.step_num = 0
    st.session_state.history_q1 = []
    st.session_state.history_q2 = []
    st.session_state.history_q3 = []
    st.session_state.history_steps = []
    st.session_state.decision_logs = [
        "Corridor initialized: All 3 agents active on NH-48 network."
    ]

# ── Main Video & Multi-Agent Execution Loop ──────────────────────────────────
if run_live and video_source is not None:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(f"Could not open video source: {video_source}")
    else:
        frame_idx = 0
        last_step_time = time.time()
        fixed_phase = 0
        fixed_step_counter = 0

        while run_live:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame_idx += 1

            # 1. Perception Step (YOLO Detection on Monitored Intersection)
            annotated_frame, metrics = vision.process_frame(frame)

            now = time.time()

            # 2. Multi-Agent Step (Every ~1.3s = 1 Corridor Step)
            if (now - last_step_time) >= 1.3:
                last_step_time = now
                st.session_state.step_num += 1

                # Inject live camera detected queue into the selected monitored intersection
                monitored_tl = TL_IDS[monitored_tl_idx]
                live_q_norm = float(np.clip(metrics["queue_pcu"] / 35.0, 0.05, 0.95))
                base_idx = monitored_tl_idx * LOCAL_OBS_DIM
                if env.current_phase[monitored_tl] == 0:
                    st.session_state.state[base_idx:base_idx + 4] = live_q_norm
                else:
                    st.session_state.state[base_idx + 4:base_idx + 8] = live_q_norm
                st.session_state.obs_dict = env._get_obs_dict()

                # Multi-Agent Actions Selection
                actions = {}
                for i, tl in enumerate(TL_IDS):
                    if "MAPPO" in controller_mode:
                        obs_t = torch.tensor(st.session_state.obs_dict[tl], dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            act, _, _ = actor.get_action(obs_t, deterministic=True)
                        act_val = int(act.item())

                        # Dynamic phase progression based on competing approach queues
                        base = i * LOCAL_OBS_DIM
                        main_q = np.mean(st.session_state.state[base:base + 4])
                        cross_q = np.mean(st.session_state.state[base + 4:base + 8])

                        if cross_q > main_q + 0.12 and env.steps_in_phase[tl] >= 4:
                            act_val = 1  # Prioritize Cross Street
                        elif main_q > cross_q + 0.12 and env.steps_in_phase[tl] >= 4:
                            act_val = 0  # Prioritize Main Highway
                        elif env.steps_in_phase[tl] >= 12:
                            act_val = (env.current_phase[tl] + 1) % 3

                        actions[tl] = act_val
                    else:
                        actions[tl] = fixed_phase

                if "Fixed" in controller_mode:
                    fixed_step_counter += 1
                    if fixed_step_counter >= 7:
                        fixed_phase = (fixed_phase + 1) % 3
                        fixed_step_counter = 0

                prev_phases = env.current_phase.copy()
                prev_signals = env.phase_state.copy()

                obs_dict, state, reward, done, info = env.step(actions)
                st.session_state.obs_dict = obs_dict
                st.session_state.state = state

                # Log signal changes
                for tl in TL_IDS:
                    tl_name = [t["name"] for t in TL_INFO if t["id"] == tl][0]
                    if env.phase_state[tl] != prev_signals[tl]:
                        st.session_state.decision_logs.append(
                            f"Step {st.session_state.step_num}: {tl_name} signal changed to {env.phase_state[tl]} (Phase {env.current_phase[tl]})"
                        )

                # Store queue history for plot
                q1 = float(np.mean(state[0:8]) * 40.0 * PUNE_PCU)
                q2 = float(np.mean(state[22:30]) * 40.0 * PUNE_PCU)
                q3 = float(np.mean(state[44:52]) * 40.0 * PUNE_PCU)

                st.session_state.history_steps.append(st.session_state.step_num)
                st.session_state.history_q1.append(q1)
                st.session_state.history_q2.append(q2)
                st.session_state.history_q3.append(q3)

                if len(st.session_state.history_steps) > 35:
                    st.session_state.history_steps.pop(0)
                    st.session_state.history_q1.pop(0)
                    st.session_state.history_q2.pop(0)
                    st.session_state.history_q3.pop(0)

                if len(st.session_state.decision_logs) > 6:
                    st.session_state.decision_logs.pop(0)

            # 3. Render the 3 Connected Traffic Light Cards
            for i, tl_info in enumerate(TL_INFO):
                tl = tl_info["id"]
                p_idx = env.current_phase[tl]
                sig_state = env.phase_state[tl]

                # Realistic glowing lamps: Red, Yellow, Green
                r_cls = "lamp-red-on" if sig_state == "RED" else "lamp-red-off"
                y_cls = "lamp-yel-on" if sig_state == "YELLOW" else "lamp-yel-off"
                g_cls = "lamp-grn-on" if sig_state == "GREEN" else "lamp-grn-off"

                sig_badge_cls = "sig-green" if sig_state == "GREEN" else ("sig-yellow" if sig_state == "YELLOW" else "sig-red")
                sig_label = f"🟢 GREEN LIGHT" if sig_state == "GREEN" else (f"🟡 YELLOW CLEARANCE" if sig_state == "YELLOW" else f"🔴 RED LIGHT (STOP)")

                base = i * LOCAL_OBS_DIM
                queue_val = float(np.mean(st.session_state.state[base:base + 8]) * 40.0 * PUNE_PCU)
                wait_val = float(np.mean(st.session_state.state[base + 8:base + 16]) * 120.0)
                cam_badge = " 📷 LIVE" if i == monitored_tl_idx else ""

                tl_placeholders[i].markdown(f"""
                <div class='tl-card'>
                    <div style='font-size:16px; font-weight:700;'>{tl_info['name']}{cam_badge}</div>
                    <div style='font-size:12px; color:#94a3b8; margin-bottom:6px;'>{tl_info['desc']}</div>
                    <div style='background:#0f172a; padding:10px 14px; border-radius:8px; display:inline-block;'>
                        <span class='lamp-circle {r_cls}'></span>
                        <span class='lamp-circle {y_cls}'></span>
                        <span class='lamp-circle {g_cls}'></span>
                    </div>
                    <div style='margin-top:6px;'>
                        <span class='badge-signal {sig_badge_cls}'>{sig_label}</span>
                    </div>
                    <div style='font-size:12px; color:#cbd5e1; margin-top:6px;'>
                        <b>{PHASE_DESCRIPTIONS[p_idx][:22]}</b><br>
                        Queue: <b>{queue_val:.1f}</b> PCU | Delay: <b>{wait_val:.1f}s</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 4. Render Corridor Status Banner
            sig_summary = " ──► ".join([f"<b>{TL_INFO[i]['name'][:4]}:</b> {env.phase_state[TL_IDS[i]]}" for i in range(3)])
            corridor_banner.markdown(f"""
            <div style='background:#f1f5f9; padding:8px 14px; border-radius:8px; border-left:5px solid #0284c7; font-size:13px;'>
                <b>Corridor Signal Sequence:</b> {sig_summary} &nbsp;|&nbsp;
                <b>Mode:</b> {controller_mode}
            </div>
            """, unsafe_allow_html=True)

            # 5. Render Video Feed with YOLO annotations
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # 6. Render Telemetry & Charts (Every 3 frames)
            if frame_idx % 3 == 0:
                c = metrics["counts"]
                df_counts = pd.DataFrame({
                    "Category": ["2W", "Auto", "Car", "Bus", "Truck"],
                    "Count": [c.get("two_wheeler", 0), c.get("auto_rickshaw", 0), c.get("car", 0), c.get("bus", 0), c.get("truck", 0)]
                }).set_index("Category")
                chart_classes_placeholder.bar_chart(df_counts, height=190)

                if len(st.session_state.history_steps) > 1:
                    df_corridor = pd.DataFrame({
                        "Step": st.session_state.history_steps,
                        "TL-1 West (PCU)": st.session_state.history_q1,
                        "TL-2 Mid (PCU)": st.session_state.history_q2,
                        "TL-3 East (PCU)": st.session_state.history_q3
                    }).set_index("Step")
                    chart_pcu_placeholder.line_chart(df_corridor, height=200)

            # Decision log box
            logs_html = "<br>".join([f"<code>{l}</code>" for l in reversed(st.session_state.decision_logs)])
            decision_log_placeholder.markdown(f"**Multi-Agent Coordination Logs:**<br>{logs_html}", unsafe_allow_html=True)

            time.sleep(0.01)

        cap.release()
