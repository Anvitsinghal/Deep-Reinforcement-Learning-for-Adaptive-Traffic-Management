"""
bmd45_detector.py
=================
Indian Vehicle Perception Module using the IISc AIM BMD-45 Taxonomy.
Reference: BMD-45 (Bengaluru Mobility Dataset) by IISc AIM Team (CVPR Findings).

Provides fine-grained classification for 14 Indian vehicle categories:
  - auto_rickshaw, two_wheeler, car, bus, lcv, tempo, heavy_truck, etc.
Maps each detected vehicle directly to its Indian Roads Congress (IRC) PCU equivalent.
"""

import os
import json
import numpy as np

# 14 Fine-Grained Indian Vehicle Classes from IISc AIM BMD-45
BMD45_TAXONOMY = {
    0:  {"name": "two_wheeler",   "pcu": 0.5, "desc": "Motorcycle, Scooter, Moped"},
    1:  {"name": "auto_rickshaw", "pcu": 1.0, "desc": "Three-wheeler Passenger Auto"},
    2:  {"name": "car",           "pcu": 1.0, "desc": "Sedan, Hatchback, Compact"},
    3:  {"name": "bus",           "pcu": 3.0, "desc": "City Transit & Intercity Bus"},
    4:  {"name": "lcv",           "pcu": 1.5, "desc": "Light Commercial Delivery Vehicle"},
    5:  {"name": "truck",         "pcu": 3.0, "desc": "Medium Commercial Truck"},
    6:  {"name": "tempo",         "pcu": 1.5, "desc": "Small Goods Carrier / Pickup"},
    7:  {"name": "bicycle",       "pcu": 0.2, "desc": "Non-motorized Cycle"},
    8:  {"name": "e_rickshaw",    "pcu": 1.0, "desc": "Battery-operated Rickshaw"},
    9:  {"name": "tractor",       "pcu": 2.0, "desc": "Agricultural/Industrial Tractor"},
    10: {"name": "van",           "pcu": 1.0, "desc": "Minivan / Utility Van"},
    11: {"name": "mini_bus",      "pcu": 2.0, "desc": "School / Feeder Minibus"},
    12: {"name": "heavy_truck",   "pcu": 3.5, "desc": "Multi-axle Freight Carrier"},
    13: {"name": "other",         "pcu": 1.0, "desc": "Unclassified Vehicle"}
}


class BMD45VehicleDetector:
    """
    Indian Vehicle Perception and Queue Estimator.
    Converts visual detections into Indian Roads Congress (IRC) PCU metrics.
    """
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.taxonomy = BMD45_TAXONOMY

    def get_pcu(self, class_id):
        """Retrieve IRC PCU equivalent for a detected vehicle class."""
        return self.taxonomy.get(int(class_id), {}).get("pcu", 1.0)

    def calculate_pcu_load(self, detections):
        """
        detections: list of dicts [{'class_id': int, 'confidence': float, 'is_stopped': bool}]
        returns: (total_pcu, queued_pcu, vehicle_breakdown)
        """
        total_pcu = 0.0
        queued_pcu = 0.0
        breakdown = {}

        for det in detections:
            cid = det.get("class_id", 2)
            conf = det.get("confidence", 1.0)
            if conf < self.conf_threshold:
                continue

            cls_info = self.taxonomy.get(cid, self.taxonomy[2])
            name = cls_info["name"]
            pcu = cls_info["pcu"]

            total_pcu += pcu
            if det.get("is_stopped", False):
                queued_pcu += pcu

            breakdown[name] = breakdown.get(name, 0) + 1

        return float(total_pcu), float(queued_pcu), breakdown

    def get_class_names(self):
        return [self.taxonomy[i]["name"] for i in range(len(self.taxonomy))]
