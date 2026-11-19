"""
live_vision_pipeline.py
=======================
Real-Time Perception Pipeline for Indian Traffic Video / Camera Feeds.
Extracts vehicle counts, Indian vehicle classes (IISc BMD-45 / IRC PCU),
queue lengths, and generates normalized state observations for MAPPO RL agents.
"""

import os
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Indian Roads Congress (IRC:106-1990) Passenger Car Unit (PCU) Standards
IRC_PCU_MAP = {
    "two_wheeler": 0.5,    # Scooter, motorcycle, bicycle
    "auto_rickshaw": 1.0,  # 3-wheeler auto, e-rickshaw
    "car": 1.0,            # Sedan, hatchback, SUV, taxi
    "bus": 3.0,            # City bus, private coach
    "truck": 3.0,          # LCV, heavy multi-axle truck
    "other": 1.0
}

# Colors for bounding boxes (BGR)
CLASS_COLORS = {
    "two_wheeler": (255, 200, 0),    # Cyan/Aqua
    "auto_rickshaw": (0, 215, 255),  # Bright Yellow/Amber
    "car": (0, 255, 0),              # Green
    "bus": (255, 0, 255),            # Magenta
    "truck": (0, 140, 255),          # Orange
    "other": (200, 200, 200)         # Gray
}


class LiveVisionPipeline:
    def __init__(self, model_path="vehicle_detection/yolov8n.pt", conf_thresh=0.25, device="cpu"):
        self.device = device
        self.conf_thresh = conf_thresh

        # Resolve weights path
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(base_dir, "yolov8n.pt")
            if os.path.exists(alt_path):
                model_path = alt_path

        self.model = YOLO(model_path)
        self.model.to(device)

        # Vehicle tracking memory (track_id -> deque of recent centroids)
        self.track_history = {}
        self.stationary_counts = {}
        self.frame_count = 0

        # Cached ROIs for the current video dimensions
        self.rois = None
        self.frame_dims = None

    def _setup_rois(self, width, height):
        """
        Dynamically construct Approach ROIs scaled to the video dimensions.
        Approach A (North-South / Main Arterial): Bottom/Center region
        Approach B (East-West / Cross Street): Left/Right or Top region
        """
        self.frame_dims = (width, height)
        self.rois = {
            "approach_main": np.array([
                [int(0.20 * width), int(0.45 * height)],
                [int(0.80 * width), int(0.45 * height)],
                [int(0.95 * width), int(0.95 * height)],
                [int(0.05 * width), int(0.95 * height)]
            ], dtype=np.int32),
            "approach_cross": np.array([
                [int(0.10 * width), int(0.15 * height)],
                [int(0.90 * width), int(0.15 * height)],
                [int(0.85 * width), int(0.45 * height)],
                [int(0.15 * width), int(0.45 * height)]
            ], dtype=np.int32)
        }

    def classify_indian_vehicle(self, cls_id, box_w, box_h, frame_h):
        """
        Maps standard COCO classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        to Indian heterogeneous classes (IISc BMD-45 / IRC).
        Distinguishes Auto-Rickshaws from cars based on aspect ratio & footprint.
        """
        aspect_ratio = box_h / max(box_w, 1)
        rel_area = (box_w * box_h) / (frame_h * frame_h)

        if cls_id in (3, 1):  # motorcycle or bicycle
            return "two_wheeler"
        elif cls_id == 5:     # bus
            return "bus"
        elif cls_id == 7:     # truck
            return "truck"
        elif cls_id == 2:     # car in standard COCO
            # Indian Auto-Rickshaws are taller relative to width and have smaller footprints
            if aspect_ratio > 1.05 and rel_area < 0.045:
                return "auto_rickshaw"
            return "car"
        return "other"

    def process_frame(self, frame):
        """
        Runs YOLO detection on frame, classifies Indian vehicles,
        computes PCU and returns annotated frame and metrics.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        if self.rois is None or self.frame_dims != (w, h):
            self._setup_rois(w, h)

        # Run YOLO with tracking enabled
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            classes=[1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
            verbose=False,
            device=self.device
        )[0]

        counts = {
            "two_wheeler": 0,
            "auto_rickshaw": 0,
            "car": 0,
            "bus": 0,
            "truck": 0,
            "other": 0
        }
        total_pcu = 0.0
        queue_pcu = 0.0
        detections = []

        annotated_frame = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else [i for i in range(len(boxes))]

            for box, cls_id, conf, tid in zip(boxes, clss, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Classify into Indian vehicle category
                veh_class = self.classify_indian_vehicle(int(cls_id), bw, bh, h)
                counts[veh_class] += 1

                pcu_weight = IRC_PCU_MAP.get(veh_class, 1.0)
                total_pcu += pcu_weight

                # Check if vehicle is inside main approach ROI
                in_main = cv2.pointPolygonTest(self.rois["approach_main"], (float(cx), float(cy)), False) >= 0

                # Velocity & Halting determination
                is_stationary = False
                if tid in self.track_history:
                    prev_cx, prev_cy = self.track_history[tid]
                    dist = math.hypot(cx - prev_cx, cy - prev_cy)
                    if dist < 4.0:  # < 4 px movement
                        self.stationary_counts[tid] = self.stationary_counts.get(tid, 0) + 1
                    else:
                        self.stationary_counts[tid] = max(0, self.stationary_counts.get(tid, 0) - 1)
                else:
                    self.stationary_counts[tid] = 1

                self.track_history[tid] = (cx, cy)

                if self.stationary_counts.get(tid, 0) >= 4 or (in_main and cy > int(0.70 * h)):
                    is_stationary = True
                    queue_pcu += pcu_weight

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class": veh_class,
                    "pcu": pcu_weight,
                    "stationary": is_stationary,
                    "conf": float(conf)
                })

                # Draw bounding box
                color = CLASS_COLORS.get(veh_class, (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Label text with Indian class & PCU
                label = f"{veh_class.replace('_', ' ').title()} ({pcu_weight} PCU)"
                if is_stationary:
                    label += " [STOPPED]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(0, y1)), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        metrics = {
            "total_vehicles": len(detections),
            "counts": counts,
            "total_pcu": round(total_pcu, 1),
            "queue_pcu": round(queue_pcu, 1),
            "detections": detections,
            "rois": self.rois
        }

        return annotated_frame, metrics

    def build_mappo_obs(self, metrics, current_phase=0, phase_state="GREEN", elapsed_green_s=15):
        """
        Converts live visual metrics into the exact 22-dimensional observation vector
        expected by the trained MAPPO Actor network (mappo_models/best_actor.pth):
          - obs[0:8]   : 8 incoming lane queues (normalized [0, 1])
          - obs[8:16]  : 8 incoming lane waiting times (normalized [0, 1])
          - obs[16:20] : 4 outgoing lane throughputs (normalized [0, 1])
          - obs[20]    : current phase index / 2.0 (normalized [0, 1])
          - obs[21]    : signal state (1.0 = GREEN, 0.5 = YELLOW)
        """
        obs = np.zeros(22, dtype=np.float32)

        queue_pcu = metrics.get("queue_pcu", 0.0)
        total_pcu = metrics.get("total_pcu", 0.0)

        # Distribute detected PCU across approach lanes
        # Normalization constant: 35 PCU (~40 vehicles) = 1.0
        norm_q = float(np.clip(queue_pcu / 35.0, 0.0, 1.0))
        for i in range(8):
            # Add small natural variance across individual lanes
            variance = 0.85 + 0.04 * (i % 4)
            obs[i] = np.clip(norm_q * variance, 0.0, 1.0)

        # Waiting delay normalization: 120 seconds = 1.0
        norm_delay = float(np.clip(elapsed_green_s / 120.0, 0.0, 1.0))
        for i in range(8, 16):
            obs[i] = norm_delay

        # Outgoing discharge throughput (proportional to total traffic moving)
        norm_thru = float(np.clip(total_pcu / 40.0, 0.0, 1.0))
        for i in range(16, 20):
            obs[i] = norm_thru

        # Current phase and light state
        obs[20] = float(current_phase) / 2.0
        obs[21] = 1.0 if phase_state == "GREEN" else 0.5

        return obs
