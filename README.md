# Drone-Based Multi-Object Tracking (VisDrone)

## Overview
This project implements a **real-time drone-based person detection and tracking pipeline** using **YOLOv8** for detection and **ByteTrack** for multi-object tracking on the **VisDrone MOT dataset**.

The system is designed to handle key aerial vision challenges such as:
- **small object detection**
- **camera ego-motion**
- **occlusions**
- **real-time inference**

---

## Pipeline
```text
Input Video / Drone Sequence
        ↓
YOLOv8 Person Detector
        ↓
Person Detections
        ↓
ByteTrack
        ↓
Tracked Persons with IDs
        ↓
Output Video / Demo UI
