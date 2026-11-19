"""
generate_indian_traffic_video.py
================================
Generates a realistic synthetic Indian intersection CCTV video clip (MP4)
featuring Indian vehicles (Two-Wheelers, Auto-Rickshaws, Cars, Buses, Trucks)
approaching an intersection with stop lines, zebra crossings, and dynamic queuing.

Used as a reliable, offline-ready test clip for live perception & signal control.
"""

import os
import random
import cv2
import numpy as np

OUTPUT_PATH = "vehicle_detection/synthetic_indian_traffic.mp4"
WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SEC = 15
TOTAL_FRAMES = FPS * DURATION_SEC


class Vehicle:
    def __init__(self, v_type, lane_x, speed, start_y):
        self.v_type = v_type
        self.x = lane_x
        self.y = start_y
        self.speed = speed
        self.target_speed = speed
        self.stopped = False

        # Dimensions & Colors by Indian Vehicle Class
        if v_type == "two_wheeler":
            self.w, self.h = 24, 48
            self.color = (30, 30, 200)      # Red / black bike
            self.label = "2-Wheeler"
        elif v_type == "auto_rickshaw":
            self.w, self.h = 42, 60
            self.color = (0, 215, 255)      # Yellow & Green Auto
            self.label = "Auto"
        elif v_type == "bus":
            self.w, self.h = 65, 160
            self.color = (30, 60, 220)      # Red State Bus
            self.label = "Bus"
        elif v_type == "truck":
            self.w, self.h = 60, 140
            self.color = (40, 120, 200)     # Orange Commercial Truck
            self.label = "Truck"
        else:  # car
            self.w, self.h = 50, 95
            self.color = (220, 220, 220)    # White / Silver car
            self.label = "Car"

    def update(self, stop_line_y, signal_red=True):
        dist_to_stop = stop_line_y - (self.y + self.h)
        if signal_red and 0 < dist_to_stop < 120:
            # Decelerate to stop line
            self.speed = max(0.0, self.speed - 0.25)
            if dist_to_stop < 10:
                self.speed = 0.0
                self.stopped = True
        else:
            self.stopped = False
            self.speed = min(self.target_speed, self.speed + 0.15)

        self.y += self.speed


def draw_road(frame, stop_line_y):
    # Dark asphalt road background
    cv2.rectangle(frame, (240, 0), (1040, HEIGHT), (45, 48, 52), -1)

    # Road edges (curbs)
    cv2.line(frame, (240, 0), (240, HEIGHT), (180, 180, 180), 4)
    cv2.line(frame, (1040, 0), (1040, HEIGHT), (180, 180, 180), 4)

    # Yellow central median / divider
    cv2.line(frame, (640, 0), (640, HEIGHT), (0, 215, 255), 4)

    # White lane dashed lines
    for lane_x in [440, 840]:
        for y in range(0, HEIGHT, 40):
            cv2.line(frame, (lane_x, y), (lane_x, y + 20), (240, 240, 240), 2)

    # Stop Line
    cv2.line(frame, (240, stop_line_y), (1040, stop_line_y), (255, 255, 255), 6)

    # Zebra crossing stripes
    for zx in range(250, 1030, 45):
        cv2.rectangle(frame, (zx, stop_line_y + 12), (zx + 25, stop_line_y + 45), (240, 240, 240), -1)


def draw_vehicle(frame, v):
    vx, vy, vw, vh = int(v.x), int(v.y), v.w, v.h

    if v.v_type == "auto_rickshaw":
        # Yellow body + Green hood (Classic Indian Auto-Rickshaw)
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh), (0, 160, 40), -1)
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh // 2), (0, 215, 255), -1)
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh), (20, 20, 20), 2)
        # Windshield
        cv2.rectangle(frame, (vx - vw // 2 + 4, vy + 4), (vx + vw // 2 - 4, vy + 14), (200, 230, 255), -1)
    elif v.v_type == "two_wheeler":
        # Motorcycle body + Rider helmet
        cv2.line(frame, (vx, vy), (vx, vy + vh), (20, 20, 20), 4)
        cv2.circle(frame, (vx, vy + vh // 3), 9, (220, 50, 50), -1)  # Helmet
        cv2.rectangle(frame, (vx - 10, vy + vh // 2), (vx + 10, vy + vh // 2 + 12), (50, 50, 180), -1)
    else:
        # Standard Car / Bus / Truck body
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh), v.color, -1)
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh), (30, 30, 30), 2)
        # Windshields (front and rear)
        cv2.rectangle(frame, (vx - vw // 2 + 4, vy + 6), (vx + vw // 2 - 4, vy + 20), (180, 210, 240), -1)
        cv2.rectangle(frame, (vx - vw // 2 + 4, vy + vh - 18), (vx + vw // 2 - 4, vy + vh - 6), (180, 210, 240), -1)

    # Headlights
    cv2.circle(frame, (vx - vw // 2 + 5, vy + 3), 3, (200, 255, 255), -1)
    cv2.circle(frame, (vx + vw // 2 - 5, vy + 3), 3, (200, 255, 255), -1)


def main():
    print(f"Generating synthetic Indian traffic video: {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    stop_line_y = int(HEIGHT * 0.72)
    lane_centers = [340, 540, 740, 940]
    classes_pool = ["two_wheeler", "auto_rickshaw", "car", "car", "bus", "truck", "two_wheeler", "auto_rickshaw"]

    vehicles = []
    # Seed initial vehicles
    for i in range(12):
        lc = random.choice(lane_centers)
        vt = random.choice(classes_pool)
        sy = random.randint(-400, stop_line_y - 120)
        sp = random.uniform(2.5, 4.5)
        vehicles.append(Vehicle(vt, lc, sp, sy))

    for frame_idx in range(TOTAL_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (70, 75, 80), dtype=np.uint8)

        # Signal state toggles: First 8 seconds RED (queuing up), then GREEN (flushing)
        signal_red = frame_idx < (FPS * 8)

        # Draw road infrastructure
        draw_road(frame, stop_line_y)

        # Spawn new vehicles from top
        if frame_idx % 25 == 0 and len(vehicles) < 22:
            lc = random.choice(lane_centers)
            vt = random.choice(classes_pool)
            sp = random.uniform(2.5, 4.5)
            vehicles.append(Vehicle(vt, lc, sp, -150))

        # Update and draw vehicles
        for v in vehicles:
            v.update(stop_line_y, signal_red=signal_red)
            draw_vehicle(frame, v)

        # Clean vehicles that moved off screen
        vehicles = [v for v in vehicles if v.y < HEIGHT + 100]

        # Environmental details (CCTV timestamp)
        sim_sec = frame_idx / FPS
        time_text = f"CCTV CAM-01 [NEW DELHI NH-48]  REC {time_text_format(sim_sec)}"
        cv2.putText(frame, time_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[OK] Synthetic Indian traffic video created at: {OUTPUT_PATH}")


def time_text_format(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 100)
    return f"14:{m:02d}:{s:02d}.{ms:02d}"


if __name__ == "__main__":
    main()
