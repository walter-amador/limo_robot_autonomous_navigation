# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CPSC-5366EL Autonomous Mobile Robotics — Assignment 4. Implements autonomous navigation for the **Agilex LIMO robot** running ROS2. The two production files are `ros2_line_follower.py` and `usb_camera_node.py`.

## Running on the Robot

```bash
# Terminal 1 — publish camera frames
python3 usb_camera_node.py

# Terminal 2 — run the line follower
python3 ros2_line_follower.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## `usb_camera_node.py`

ROS2 node that captures from a USB camera and publishes to two topics:
- `/camera/image_raw` (`sensor_msgs/Image`, bgr8)
- `/camera/camera_info` (`sensor_msgs/CameraInfo`)

Parameters (set via ROS2 param or defaults): `video_device` (`/dev/video1`), `width` (640), `height` (480), `fps` (60).

## `ros2_line_follower.py`

The main autonomous navigation node. Subscribes to `/camera/image_raw` and publishes `geometry_msgs/Twist` to `/cmd_vel`.

**Processing pipeline per frame:**

1. **Lane detection** — HSV threshold (green: `[30,90,25]–[85,250,255]`) → morphology → largest contour → centroid → PD error signal.
2. **Sign detection** — YOLOv8n (`model/road_signs_yolov8n.pt`) detects stop, no-entry, dead-end, and junction signs; detections drive FSM transitions.
3. **Obstacle detection** — HSV threshold (orange cones: `CONE_LOWER_HSV`/`CONE_UPPER_HSV`) within a trapezoidal ROI; proximity is determined by normalized Y-position in frame.
4. **FSM navigation** — states: `FOLLOW_LANE`, `STOP_WAIT`, `TURN_180`.
5. **PD control** — outputs `angular.z`; `linear.x` is a fixed speed reduced during avoidance.

**Obstacle avoidance** overlays a 5-phase sequence on top of the FSM:
- Phase 1: Turn away from the obstacle side
- Phase 2: Move forward past the obstacle
- Phase 3: Turn back toward the lane
- Phase 4: Forward clearance (distance = `ROBOT_WIDTH/2 + CONE_WIDTH/2 + CLEARANCE_SAFETY_MARGIN` ≈ 0.23 m)
- Phase 5: Re-center on lane

Phase timeouts (`PHASE_TIMEOUT_TURN`, `PHASE_TIMEOUT_FORWARD`) prevent getting stuck. `PHASE1_MAX_RESTARTS` forces a sustained turn when the obstacle persists after repeated restarts.

**QoS:** `BEST_EFFORT` + `VOLATILE` + `KEEP_LAST` depth=1 to match the camera publisher.

**Key tunable constants** (all instance variables in `__init__`):

| Constant | Meaning |
|----------|---------|
| `LOWER_HSV` / `UPPER_HSV` | Green lane HSV range |
| `CONE_LOWER_HSV` / `CONE_UPPER_HSV` | Orange cone HSV range |
| `KP`, `KD` | Lane-following PD gains |
| `KP_AVOIDANCE`, `KD_AVOIDANCE` | Avoidance PD gains |
| `LINEAR_SPEED` | Base forward speed (m/s) |
| `OBSTACLE_DANGER_ZONE_Y` | Normalized Y threshold for immediate avoidance |

## Other Files

| Path | Purpose |
|------|---------|
| `desktop_sims/vid_sim.py` | Full pipeline simulation on a video file (no ROS2) |
| `desktop_sims/vslam_sim.py` | Visual SLAM simulation (ORB + Bag of Visual Words loop closure) |
| `desktop_sims/img_sim.py` | Single-image lane detection sanity check |
| `desktop_sims/vid_rec.py` | Video playback / frame-save tool for reviewing recordings |
| `desktop_sims/depth.py` | MiDaS monocular depth estimation experiment |
| `desktop_sims/hsv_tuner.py` | Interactive trackbar UI for tuning HSV ranges |
| `robot_experiments/avoidance.py` | Standalone obstacle avoidance node (early prototype) |
| `robot_experiments/ros2_line_follower_fastsam.py` | Variant using FastSAM instead of HSV for obstacle detection |
| `robot_experiments/ros2_line_obs_det.py` | Obstacle detection-only node (no sign handling) |
| `robot_experiments/ros2_vslam.py` | Visual SLAM ROS2 node (ORB features + wheel odometry fusion) |
| `model/road_signs_yolov8n.pt` | Custom YOLOv8n for road sign detection |
| `model/FastSAM-s.pt` | FastSAM-s for general obstacle segmentation |
| `imgs/` | Saved frames for offline HSV/contour testing |
