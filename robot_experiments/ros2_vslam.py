#!/usr/bin/env python3

"""
ROS2 Visual SLAM Node for AgileX LIMO Robot
============================================

This implementation provides a 2D visual odometry and mapping system for the
AgileX LIMO robot using a single (monocular) camera with wheel odometry fusion.

SUBSCRIPTIONS:
- /camera/image_raw (sensor_msgs/Image): Camera feed
- /odom (nav_msgs/Odometry): Wheel odometry for scale estimation

PUBLICATIONS:
- /vslam/pose (geometry_msgs/PoseStamped): Current estimated pose
- /vslam/map_image (sensor_msgs/Image): 2D map visualization
- /vslam/debug_image (sensor_msgs/Image): Annotated camera view
- /vslam/status (std_msgs/String): VSLAM status information

KEY FEATURES:
- Feature-based Visual Odometry using ORB features
- Wheel odometry fusion for accurate scale estimation
- Loop closure detection using Bag of Visual Words (BoVW)
- Kidnapped robot relocalization
- Real-time 2D map visualization

Author: CPSC-5366EL Autonomous Mobile Robotics
Date: December 2025
"""

import cv2
import numpy as np
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set
import pickle
import os
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.distance import cdist
import threading

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import json


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class VSLAMConfig:
    """Configuration parameters for the VSLAM system."""

    # Feature Detection
    n_features: int = 1000  # Number of ORB features to detect
    scale_factor: float = 1.2  # Pyramid scale factor
    n_levels: int = 8  # Number of pyramid levels

    # Feature Matching
    match_ratio: float = 0.75  # Lowe's ratio test threshold
    min_matches: int = 50  # Minimum matches for valid motion estimation

    # Keyframe Selection
    keyframe_min_distance: float = 0.3  # Minimum distance (meters) for new keyframe
    keyframe_min_rotation: float = 0.15  # Minimum rotation (radians) for new keyframe
    keyframe_min_matches: int = 100  # Minimum feature matches to consider same place

    # Loop Closure
    loop_closure_threshold: float = 0.6  # Similarity threshold for loop closure
    loop_closure_min_frames: int = 30  # Minimum frames since last keyframe to check

    # Relocalization (Kidnapped Robot)
    relocalization_threshold: float = 0.5  # Similarity threshold for relocalization
    lost_tracking_frames: int = 10  # Frames without matches = lost

    # Camera Intrinsics (LIMO robot camera - should be calibrated)
    focal_length: float = 500.0  # Focal length in pixels
    principal_point: Tuple[float, float] = (320.0, 240.0)  # Principal point (cx, cy)

    # Map Visualization
    map_scale: float = 100.0  # Pixels per meter in the 2D map
    map_size: Tuple[int, int] = (800, 800)  # Map canvas size

    # Odometry fusion
    use_wheel_odometry: bool = True  # Use wheel odometry for scale
    odom_scale_weight: float = 0.8  # Weight for wheel odometry scale (vs visual)


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================


class FeatureExtractor:
    """
    Extracts ORB features from images.

    ORB (Oriented FAST and Rotated BRIEF) is chosen because:
    - Fast computation (suitable for real-time robotics)
    - Rotation invariant
    - Scale invariant through pyramid
    - Binary descriptors = fast matching
    """

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.n_features,
            scaleFactor=config.scale_factor,
            nlevels=config.n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        Args:
            image: Input BGR or grayscale image

        Returns:
            keypoints: List of detected keypoints
            descriptors: Numpy array of descriptors (N x 32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        return keypoints, descriptors


# =============================================================================
# FEATURE MATCHER
# =============================================================================


class FeatureMatcher:
    """
    Matches ORB features between frames using FLANN-based matching.

    Uses LSH (Locality Sensitive Hashing) index for binary descriptors,
    which is efficient for ORB's binary descriptors.
    """

    def __init__(self, config: VSLAMConfig):
        self.config = config

        # FLANN parameters for binary descriptors (ORB)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two frames using ratio test.

        Args:
            desc1: Descriptors from frame 1
            desc2: Descriptors from frame 2

        Returns:
            matches: List of good matches after ratio test
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio * n.distance:
                    good_matches.append(m)

        return good_matches


# =============================================================================
# BAG OF VISUAL WORDS (BoVW) - VISUAL VOCABULARY
# =============================================================================


class VisualVocabulary:
    """
    Bag of Visual Words implementation for efficient place recognition.
    """

    def __init__(self, vocabulary_size: int = 500, min_samples_for_training: int = 30):
        self.vocabulary_size = vocabulary_size
        self.min_samples_for_training = min_samples_for_training

        self.vocabulary: Optional[np.ndarray] = None
        self.is_trained = False

        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.document_frequency: Dict[int, int] = defaultdict(int)
        self.bow_vectors: Dict[int, np.ndarray] = {}
        self.num_keyframes = 0
        self.training_descriptors: List[np.ndarray] = []

    def _convert_binary_to_float(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            return np.array([])
        float_desc = np.unpackbits(descriptors, axis=1).astype(np.float32)
        return float_desc

    def add_training_sample(self, descriptors: np.ndarray):
        if descriptors is not None and len(descriptors) > 0:
            self.training_descriptors.append(descriptors.copy())

    def train_vocabulary(self, descriptors_list: List[np.ndarray] = None):
        if descriptors_list is None:
            descriptors_list = self.training_descriptors

        if len(descriptors_list) == 0:
            return False

        all_descriptors = np.vstack(descriptors_list)

        if len(all_descriptors) < self.vocabulary_size:
            return False

        float_descriptors = self._convert_binary_to_float(all_descriptors)

        max_samples = 10000
        if len(float_descriptors) > max_samples:
            indices = np.random.choice(
                len(float_descriptors), max_samples, replace=False
            )
            float_descriptors = float_descriptors[indices]

        try:
            centroids, labels = kmeans2(
                float_descriptors,
                self.vocabulary_size,
                minit="random",
                iter=10,
                missing="warn",
            )

            self.vocabulary = centroids
            self.is_trained = True
            self.training_descriptors = []
            return True

        except Exception as e:
            return False

    def _quantize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        if not self.is_trained or descriptors is None or len(descriptors) == 0:
            return np.array([])

        float_desc = self._convert_binary_to_float(descriptors)
        word_ids, _ = vq(float_desc, self.vocabulary)
        return word_ids

    def compute_bow_vector(
        self, descriptors: np.ndarray, use_tfidf: bool = True
    ) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(self.vocabulary_size)

        word_ids = self._quantize_descriptors(descriptors)

        if len(word_ids) == 0:
            return np.zeros(self.vocabulary_size)

        bow = np.bincount(word_ids, minlength=self.vocabulary_size).astype(np.float64)

        if bow.sum() > 0:
            bow = bow / bow.sum()

        if use_tfidf and self.num_keyframes > 0:
            for word_id in range(self.vocabulary_size):
                df = self.document_frequency.get(word_id, 0)
                if df > 0:
                    idf = np.log(self.num_keyframes / df)
                    bow[word_id] *= idf

        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm

        return bow

    def add_keyframe(self, keyframe_id: int, descriptors: np.ndarray):
        if not self.is_trained:
            self.add_training_sample(descriptors)
            return

        if descriptors is None or len(descriptors) == 0:
            return

        word_ids = self._quantize_descriptors(descriptors)

        if len(word_ids) == 0:
            return

        word_counts = np.bincount(word_ids, minlength=self.vocabulary_size)
        total_words = len(word_ids)

        unique_words = np.where(word_counts > 0)[0]
        for word_id in unique_words:
            tf = word_counts[word_id] / total_words
            self.inverted_index[word_id].append((keyframe_id, tf))
            self.document_frequency[word_id] += 1

        self.bow_vectors[keyframe_id] = self.compute_bow_vector(
            descriptors, use_tfidf=False
        )
        self.num_keyframes += 1

    def query(
        self, descriptors: np.ndarray, top_k: int = 10, exclude_recent: int = 0
    ) -> List[Tuple[int, float]]:
        if not self.is_trained or descriptors is None or len(descriptors) == 0:
            return []

        query_bow = self.compute_bow_vector(descriptors)

        if np.linalg.norm(query_bow) == 0:
            return []

        word_ids = self._quantize_descriptors(descriptors)
        unique_words = np.unique(word_ids)

        candidate_scores: Dict[int, float] = defaultdict(float)

        for word_id in unique_words:
            if word_id not in self.inverted_index:
                continue

            df = self.document_frequency.get(word_id, 1)
            idf = np.log(max(self.num_keyframes, 1) / df) if df > 0 else 0
            query_tf = np.sum(word_ids == word_id) / len(word_ids)

            for kf_id, kf_tf in self.inverted_index[word_id]:
                if self.num_keyframes - kf_id <= exclude_recent:
                    continue
                candidate_scores[kf_id] += query_tf * kf_tf * (idf**2)

        if len(candidate_scores) == 0:
            return []

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        top_candidates = sorted_candidates[: min(top_k * 3, len(sorted_candidates))]

        results = []
        for kf_id, _ in top_candidates:
            if kf_id in self.bow_vectors:
                kf_bow = self.bow_vectors[kf_id]
                similarity = np.dot(query_bow, kf_bow)
                results.append((kf_id, similarity))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]


# =============================================================================
# VISUAL ODOMETRY WITH WHEEL ODOMETRY FUSION
# =============================================================================


class VisualOdometry:
    """
    Estimates camera motion between consecutive frames with wheel odometry fusion.
    """

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.matcher = FeatureMatcher(config)

        # Camera intrinsic matrix
        self.K = np.array(
            [
                [config.focal_length, 0, config.principal_point[0]],
                [0, config.focal_length, config.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_image = None

        # Current pose (accumulated)
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.eye(3)  # Rotation matrix

        # Wheel odometry data
        self.prev_odom_position = None
        self.prev_odom_orientation = None
        self.odom_delta_distance = 0.0  # Distance moved since last frame

        # Tracking statistics
        self.num_matches = 0
        self.tracking_quality = 1.0
        self.frames_without_matches = 0

    def update_wheel_odometry(self, odom_position: np.ndarray, odom_orientation: np.ndarray):
        """
        Update wheel odometry data for scale estimation.

        Args:
            odom_position: Position from wheel odometry [x, y, z]
            odom_orientation: Orientation quaternion [x, y, z, w]
        """
        if self.prev_odom_position is not None:
            # Calculate distance moved since last odometry update
            delta = odom_position - self.prev_odom_position
            self.odom_delta_distance = np.linalg.norm(delta[:2])  # 2D distance
        else:
            self.odom_delta_distance = 0.0

        self.prev_odom_position = odom_position.copy()
        self.prev_odom_orientation = odom_orientation.copy()

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a new frame and estimate motion.

        Returns:
            position: Current 3D position [x, y, z]
            orientation: Current 3x3 rotation matrix
            num_matches: Number of feature matches found
        """
        # Extract features
        keypoints, descriptors = self.extractor.extract(image)

        # If first frame, just store and return
        if self.prev_descriptors is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_image = image.copy()
            return self.position.copy(), self.orientation.copy(), 0

        # Match features
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        self.num_matches = len(matches)

        # Check if we have enough matches
        if len(matches) < self.config.min_matches:
            self.frames_without_matches += 1
            self.tracking_quality = max(0.0, self.tracking_quality - 0.1)

            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_image = image.copy()

            return self.position.copy(), self.orientation.copy(), len(matches)

        # Reset tracking loss counter
        self.frames_without_matches = 0
        self.tracking_quality = min(1.0, self.tracking_quality + 0.05)

        # Extract matched point coordinates
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Estimate Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_image = image.copy()
            return self.position.copy(), self.orientation.copy(), len(matches)

        # Recover pose from Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        # Determine scale from wheel odometry or use default
        if self.config.use_wheel_odometry and self.odom_delta_distance > 0.001:
            # Use wheel odometry for scale (much more accurate than visual-only)
            scale = self.odom_delta_distance
        else:
            # Fallback to a small default scale
            scale = 0.05

        # Update pose using dead reckoning
        t_world = self.orientation @ (t.flatten() * scale)
        self.position += t_world
        self.orientation = R @ self.orientation

        # Reset odometry delta after using it
        self.odom_delta_distance = 0.0

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_image = image.copy()

        return self.position.copy(), self.orientation.copy(), len(matches)

    def is_lost(self) -> bool:
        return self.frames_without_matches >= self.config.lost_tracking_frames

    def reset_pose(self, position: np.ndarray, orientation: np.ndarray):
        self.position = position.copy()
        self.orientation = orientation.copy()
        self.frames_without_matches = 0
        self.tracking_quality = 1.0


# =============================================================================
# KEYFRAME DATABASE
# =============================================================================


@dataclass
class Keyframe:
    """Stores information about a keyframe."""

    id: int
    timestamp: float
    position: np.ndarray
    orientation: np.ndarray
    descriptors: np.ndarray
    keypoints: List[cv2.KeyPoint]
    thumbnail: np.ndarray = None

    def __post_init__(self):
        self.keypoints_data = [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in self.keypoints
        ]


class KeyframeDatabase:
    """Database of keyframes for loop closure and relocalization."""

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.keyframes: List[Keyframe] = []
        self.matcher = FeatureMatcher(config)
        self.next_id = 0

        self.vocabulary = VisualVocabulary(
            vocabulary_size=500,
            min_samples_for_training=30,
        )

        self.vocab_trained_at_keyframe = -1

    def should_add_keyframe(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> bool:
        if len(self.keyframes) == 0:
            return True

        last_kf = self.keyframes[-1]

        distance = np.linalg.norm(position - last_kf.position)
        if distance >= self.config.keyframe_min_distance:
            return True

        R_diff = orientation @ last_kf.orientation.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        if angle >= self.config.keyframe_min_rotation:
            return True

        return False

    def add_keyframe(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        descriptors: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        image: np.ndarray = None,
    ):
        thumbnail = None
        if image is not None:
            thumbnail = cv2.resize(image, (160, 120))

        kf = Keyframe(
            id=self.next_id,
            timestamp=time.time(),
            position=position.copy(),
            orientation=orientation.copy(),
            descriptors=descriptors.copy() if descriptors is not None else None,
            keypoints=list(keypoints) if keypoints else [],
            thumbnail=thumbnail,
        )

        self.keyframes.append(kf)

        if descriptors is not None:
            if not self.vocabulary.is_trained:
                self.vocabulary.add_training_sample(descriptors)

                if (
                    len(self.vocabulary.training_descriptors)
                    >= self.vocabulary.min_samples_for_training
                ):
                    if self.vocabulary.train_vocabulary():
                        self.vocab_trained_at_keyframe = self.next_id
                        self._reindex_all_keyframes()
            else:
                self.vocabulary.add_keyframe(self.next_id, descriptors)

        self.next_id += 1
        return kf.id

    def _reindex_all_keyframes(self):
        for kf in self.keyframes:
            if kf.descriptors is not None:
                self.vocabulary.add_keyframe(kf.id, kf.descriptors)

    def find_best_match(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        if descriptors is None or len(self.keyframes) == 0:
            return None, 0.0, 0

        if self.vocabulary.is_trained:
            return self._find_best_match_bovw(descriptors, exclude_recent)
        else:
            return self._find_best_match_bruteforce(descriptors, exclude_recent)

    def _find_best_match_bovw(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        candidates = self.vocabulary.query(
            descriptors, top_k=5, exclude_recent=exclude_recent
        )

        if len(candidates) == 0:
            return None, 0.0, 0

        best_kf_id, best_score = candidates[0]

        best_keyframe = None
        for kf in self.keyframes:
            if kf.id == best_kf_id:
                best_keyframe = kf
                break

        if best_keyframe is None:
            return None, 0.0, 0

        pseudo_matches = int(best_score * 100)
        return best_keyframe, best_score, pseudo_matches

    def _find_best_match_bruteforce(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        best_keyframe = None
        best_score = 0.0
        best_matches = 0

        search_range = len(self.keyframes) - exclude_recent

        for i in range(search_range):
            kf = self.keyframes[i]
            if kf.descriptors is None:
                continue

            matches = self.matcher.match(descriptors, kf.descriptors)
            num_matches = len(matches)

            max_possible = min(len(descriptors), len(kf.descriptors))
            if max_possible > 0:
                score = num_matches / max_possible
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_keyframe = kf
                best_matches = num_matches

        return best_keyframe, best_score, best_matches

    def get_trajectory(self) -> np.ndarray:
        if len(self.keyframes) == 0:
            return np.array([])
        return np.array([kf.position for kf in self.keyframes])


# =============================================================================
# LOOP CLOSURE DETECTOR
# =============================================================================


class LoopClosureDetector:
    """Detects when the robot returns to a previously visited location."""

    def __init__(self, config: VSLAMConfig, keyframe_db: KeyframeDatabase):
        self.config = config
        self.keyframe_db = keyframe_db
        self.matcher = FeatureMatcher(config)
        self.detected_loops: List[Tuple[int, int, float]] = []

    def check_loop_closure(
        self, descriptors: np.ndarray, current_kf_id: int
    ) -> Tuple[bool, Optional[Keyframe], float]:
        best_kf, score, num_matches = self.keyframe_db.find_best_match(
            descriptors, exclude_recent=self.config.loop_closure_min_frames
        )

        if best_kf is None:
            return False, None, 0.0

        if score >= self.config.loop_closure_threshold:
            self.detected_loops.append((current_kf_id, best_kf.id, score))
            return True, best_kf, score

        return False, None, score


# =============================================================================
# 2D MAP VISUALIZATION
# =============================================================================


class Map2D:
    """2D visualization of robot trajectory and mapping."""

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.width, self.height = config.map_size
        self.scale = config.map_scale

        self.origin = np.array([self.width // 2, self.height // 2])

        self.trajectory: List[np.ndarray] = []
        self.keyframe_positions: List[np.ndarray] = []
        self.loop_closures: List[Tuple[np.ndarray, np.ndarray]] = []

    def world_to_map(self, position: np.ndarray) -> Tuple[int, int]:
        map_x = int(self.origin[0] + position[0] * self.scale)
        map_y = int(self.origin[1] - position[2] * self.scale)
        return map_x, map_y

    def add_position(self, position: np.ndarray):
        self.trajectory.append(position.copy())

    def add_keyframe(self, position: np.ndarray):
        self.keyframe_positions.append(position.copy())

    def add_loop_closure(self, pos1: np.ndarray, pos2: np.ndarray):
        self.loop_closures.append((pos1.copy(), pos2.copy()))

    def refresh_trajectory(self, keyframe_positions: List[np.ndarray]):
        self.keyframe_positions = [pos.copy() for pos in keyframe_positions]
        self.trajectory = [pos.copy() for pos in keyframe_positions]

    def render(
        self, current_position: np.ndarray, current_orientation: np.ndarray
    ) -> np.ndarray:
        map_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        map_img[:] = (40, 40, 40)

        # Draw grid
        grid_spacing = int(1.0 * self.scale)
        for x in range(0, self.width, grid_spacing):
            cv2.line(map_img, (x, 0), (x, self.height), (60, 60, 60), 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(map_img, (0, y), (self.width, y), (60, 60, 60), 1)

        # Draw origin
        cv2.circle(map_img, tuple(self.origin), 10, (100, 100, 100), 2)
        cv2.putText(
            map_img,
            "START",
            (self.origin[0] + 15, self.origin[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
        )

        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                pt1 = self.world_to_map(self.trajectory[i - 1])
                pt2 = self.world_to_map(self.trajectory[i])
                if (
                    0 <= pt1[0] < self.width
                    and 0 <= pt1[1] < self.height
                    and 0 <= pt2[0] < self.width
                    and 0 <= pt2[1] < self.height
                ):
                    cv2.line(map_img, pt1, pt2, (255, 200, 0), 2)

        # Draw keyframes
        for kf_pos in self.keyframe_positions:
            pt = self.world_to_map(kf_pos)
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height:
                cv2.circle(map_img, pt, 5, (0, 255, 0), -1)

        # Draw loop closures
        for pos1, pos2 in self.loop_closures:
            pt1 = self.world_to_map(pos1)
            pt2 = self.world_to_map(pos2)
            if (
                0 <= pt1[0] < self.width
                and 0 <= pt1[1] < self.height
                and 0 <= pt2[0] < self.width
                and 0 <= pt2[1] < self.height
            ):
                cv2.line(map_img, pt1, pt2, (255, 0, 255), 2)

        # Draw robot
        robot_pt = self.world_to_map(current_position)
        if 0 <= robot_pt[0] < self.width and 0 <= robot_pt[1] < self.height:
            cv2.circle(map_img, robot_pt, 8, (0, 0, 255), -1)

            forward = current_orientation[:, 2]
            arrow_length = 30
            heading_end = (
                int(robot_pt[0] + forward[0] * arrow_length),
                int(robot_pt[1] - forward[2] * arrow_length),
            )
            cv2.arrowedLine(
                map_img, robot_pt, heading_end, (0, 0, 255), 2, tipLength=0.3
            )

        # Legend
        legend_y = 30
        cv2.putText(
            map_img,
            "VSLAM Map",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.circle(map_img, (20, legend_y + 25), 5, (255, 200, 0), -1)
        cv2.putText(
            map_img,
            "Trajectory",
            (35, legend_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.circle(map_img, (20, legend_y + 45), 5, (0, 255, 0), -1)
        cv2.putText(
            map_img,
            "Keyframes",
            (35, legend_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.circle(map_img, (20, legend_y + 65), 5, (0, 0, 255), -1)
        cv2.putText(
            map_img,
            "Robot",
            (35, legend_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.line(map_img, (10, legend_y + 85), (30, legend_y + 85), (255, 0, 255), 2)
        cv2.putText(
            map_img,
            "Loop Closure",
            (35, legend_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Scale
        cv2.line(
            map_img,
            (self.width - 120, self.height - 30),
            (self.width - 120 + int(self.scale), self.height - 30),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            map_img,
            "1m",
            (self.width - 95, self.height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return map_img


# =============================================================================
# MAIN VSLAM SYSTEM
# =============================================================================


class VSLAMSystem:
    """Main Visual SLAM system integrating all components."""

    def __init__(self, config: VSLAMConfig = None):
        self.config = config or VSLAMConfig()

        self.visual_odometry = VisualOdometry(self.config)
        self.keyframe_db = KeyframeDatabase(self.config)
        self.loop_detector = LoopClosureDetector(self.config, self.keyframe_db)
        self.map_2d = Map2D(self.config)
        self.feature_extractor = FeatureExtractor(self.config)

        self.frame_count = 0
        self.current_keyframe_id = -1
        self.is_relocalized = False
        self.relocalization_source = None

        self.stats = {
            "total_frames": 0,
            "keyframes": 0,
            "loop_closures": 0,
            "relocalizations": 0,
            "tracking_quality": 1.0,
        }

    def update_wheel_odometry(self, position: np.ndarray, orientation: np.ndarray):
        """Pass wheel odometry to visual odometry for scale estimation."""
        self.visual_odometry.update_wheel_odometry(position, orientation)

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame through the VSLAM pipeline."""
        self.frame_count += 1
        self.stats["total_frames"] = self.frame_count

        result = {
            "position": None,
            "orientation": None,
            "is_keyframe": False,
            "loop_closure": None,
            "is_lost": False,
            "relocalized": False,
            "map_image": None,
            "annotated_frame": None,
        }

        # Visual Odometry
        position, orientation, num_matches = self.visual_odometry.process_frame(frame)
        result["position"] = position
        result["orientation"] = orientation
        self.stats["tracking_quality"] = self.visual_odometry.tracking_quality

        self.map_2d.add_position(position)

        # Check if lost
        if self.visual_odometry.is_lost():
            result["is_lost"] = True

            keypoints, descriptors = self.feature_extractor.extract(frame)
            best_kf, score, matches = self.keyframe_db.find_best_match(descriptors)

            if best_kf is not None and score >= self.config.relocalization_threshold:
                self.visual_odometry.reset_pose(best_kf.position, best_kf.orientation)
                result["position"] = best_kf.position.copy()
                result["orientation"] = best_kf.orientation.copy()
                result["relocalized"] = True
                self.is_relocalized = True
                self.relocalization_source = best_kf.id
                self.stats["relocalizations"] += 1

        # Keyframe selection
        keypoints, descriptors = self.feature_extractor.extract(frame)

        if self.keyframe_db.should_add_keyframe(position, orientation):
            kf_id = self.keyframe_db.add_keyframe(
                position, orientation, descriptors, keypoints, frame
            )
            self.current_keyframe_id = kf_id
            result["is_keyframe"] = True
            self.stats["keyframes"] = len(self.keyframe_db.keyframes)
            self.map_2d.add_keyframe(position)

            # Loop closure
            if len(self.keyframe_db.keyframes) > self.config.loop_closure_min_frames:
                is_loop, matched_kf, confidence = self.loop_detector.check_loop_closure(
                    descriptors, kf_id
                )

                if is_loop:
                    position_error_before = np.linalg.norm(
                        position - matched_kf.position
                    )

                    result["loop_closure"] = {
                        "matched_keyframe": matched_kf.id,
                        "confidence": confidence,
                        "position_error": position_error_before,
                    }
                    self.stats["loop_closures"] += 1

                    self.map_2d.add_loop_closure(position, matched_kf.position)

                    corrected_pos = self.perform_linear_correction(position, matched_kf)

                    self.visual_odometry.position = corrected_pos.copy()
                    self.keyframe_db.keyframes[kf_id].position = corrected_pos.copy()

                    position = corrected_pos
                    result["position"] = position

                    corrected_kf_positions = [
                        kf.position for kf in self.keyframe_db.keyframes
                    ]
                    self.map_2d.refresh_trajectory(corrected_kf_positions)

        # Render
        result["map_image"] = self.map_2d.render(position, orientation)
        result["annotated_frame"] = self._annotate_frame(frame, result, num_matches)

        return result

    def _annotate_frame(
        self, frame: np.ndarray, result: Dict, num_matches: int
    ) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        keypoints, _ = self.feature_extractor.extract(frame)
        for kp in keypoints[:100]:
            pt = tuple(map(int, kp.pt))
            cv2.circle(annotated, pt, 3, (0, 255, 255), 1)

        cv2.rectangle(annotated, (5, 5), (250, 160), (0, 0, 0), -1)
        cv2.rectangle(annotated, (5, 5), (250, 160), (255, 255, 255), 1)

        pos = result["position"]
        cv2.putText(
            annotated,
            f"Position:",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"  X: {pos[0]:.2f}m",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            annotated,
            f"  Y: {pos[1]:.2f}m",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            annotated,
            f"  Z: {pos[2]:.2f}m",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            annotated,
            f"Matches: {num_matches}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if num_matches > 50 else (0, 0, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"Keyframes: {self.stats['keyframes']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"Loop Closures: {self.stats['loop_closures']}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
        )

        quality = self.stats["tracking_quality"]
        bar_width = int(100 * quality)
        bar_color = (
            (0, 255, 0)
            if quality > 0.7
            else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
        )
        cv2.rectangle(annotated, (10, 140), (10 + bar_width, 155), bar_color, -1)
        cv2.rectangle(annotated, (10, 140), (110, 155), (255, 255, 255), 1)
        cv2.putText(
            annotated,
            "Quality",
            (120, 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        if result["is_lost"]:
            cv2.putText(
                annotated,
                "TRACKING LOST!",
                (width // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if result["relocalized"]:
            cv2.putText(
                annotated,
                "RELOCALIZED!",
                (width // 2 - 80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        if result["is_keyframe"]:
            cv2.putText(
                annotated,
                "KEYFRAME",
                (width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if result["loop_closure"]:
            cv2.putText(
                annotated,
                f"LOOP CLOSURE!",
                (width // 2 - 80, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

        return annotated

    def perform_linear_correction(
        self, current_pos: np.ndarray, matched_kf: Keyframe
    ) -> np.ndarray:
        matched_kf_id = matched_kf.id
        current_kf_id = self.current_keyframe_id

        if current_kf_id <= matched_kf_id:
            return current_pos.copy()

        drift_vector = current_pos - matched_kf.position
        num_keyframes_to_correct = current_kf_id - matched_kf_id

        if num_keyframes_to_correct <= 0:
            return current_pos.copy()

        for i, kf in enumerate(self.keyframe_db.keyframes):
            if kf.id <= matched_kf_id:
                continue
            if kf.id > current_kf_id:
                continue

            alpha = (kf.id - matched_kf_id) / num_keyframes_to_correct
            correction = drift_vector * alpha
            kf.position = kf.position - correction

        corrected_pos = current_pos - drift_vector
        return corrected_pos


# =============================================================================
# ROS2 NODE
# =============================================================================


class VSLAMNode(Node):
    """
    ROS2 Node for Visual SLAM on AgileX LIMO robot.

    Subscribes to:
        /camera/image_raw - Camera feed
        /odom - Wheel odometry

    Publishes:
        /vslam/pose - Current pose estimate
        /vslam/map_image - 2D map visualization
        /vslam/debug_image - Annotated camera view
        /vslam/status - VSLAM status JSON
    """

    def __init__(self):
        super().__init__("vslam_node")

        # CV Bridge
        self.bridge = CvBridge()

        # Configuration
        config = VSLAMConfig(
            n_features=1000,
            match_ratio=0.75,
            min_matches=30,
            keyframe_min_distance=0.2,
            keyframe_min_rotation=0.1,
            loop_closure_threshold=0.5,
            map_scale=100.0,
            use_wheel_odometry=True,
        )

        # Initialize VSLAM
        self.vslam = VSLAMSystem(config)

        # State
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.last_process_time = time.time()
        self.fps = 0.0

        # Display option
        self.show_visualization = True

        # QoS for camera (sensor data)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, sensor_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, sensor_qos
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/vslam/pose", 10)
        self.map_pub = self.create_publisher(Image, "/vslam/map_image", 10)
        self.debug_pub = self.create_publisher(Image, "/vslam/debug_image", 10)
        self.status_pub = self.create_publisher(String, "/vslam/status", 10)

        # Timer for processing (20 Hz)
        self.process_timer = self.create_timer(0.05, self.process_callback)

        self.get_logger().info("=" * 60)
        self.get_logger().info("VSLAM Node Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Subscribing to: /camera/image_raw")
        self.get_logger().info("Subscribing to: /odom")
        self.get_logger().info("Publishing to: /vslam/pose, /vslam/map_image, /vslam/debug_image")
        self.get_logger().info("Press 'q' in visualization window to quit")
        self.get_logger().info("=" * 60)

    def image_callback(self, msg: Image):
        """Handle incoming camera images."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 480))

            with self.frame_lock:
                self.latest_frame = frame.copy()

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def odom_callback(self, msg: Odometry):
        """Handle incoming wheel odometry."""
        try:
            # Extract position
            position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])

            # Extract orientation (quaternion)
            orientation = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ])

            # Update VSLAM with odometry
            self.vslam.update_wheel_odometry(position, orientation)

        except Exception as e:
            self.get_logger().error(f"Failed to process odometry: {e}")

    def process_callback(self):
        """Process frames and publish results."""
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_process_time
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self.last_process_time = current_time

        # Process frame through VSLAM
        result = self.vslam.process_frame(frame)

        # Add FPS to annotated frame
        cv2.putText(
            result["annotated_frame"],
            f"FPS: {self.fps:.1f}",
            (540, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Publish pose
        self.publish_pose(result)

        # Publish images
        self.publish_images(result)

        # Publish status
        self.publish_status(result)

        # Show visualization (optional)
        if self.show_visualization:
            cv2.imshow("VSLAM - Camera View", result["annotated_frame"])
            cv2.imshow("VSLAM - 2D Map", result["map_image"])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("Quit requested, shutting down...")
                rclpy.shutdown()
            elif key == ord("r"):
                self.get_logger().info("Resetting VSLAM...")
                config = self.vslam.config
                self.vslam = VSLAMSystem(config)

        # Log events
        if result["is_keyframe"]:
            self.get_logger().info(
                f"Keyframe {self.vslam.stats['keyframes']} added at "
                f"[{result['position'][0]:.2f}, {result['position'][2]:.2f}]"
            )

        if result["loop_closure"]:
            lc = result["loop_closure"]
            self.get_logger().info(
                f"LOOP CLOSURE with KF {lc['matched_keyframe']} "
                f"(conf: {lc['confidence']:.2f}, drift: {lc['position_error']:.3f}m)"
            )

        if result["relocalized"]:
            self.get_logger().info("RELOCALIZED successfully!")

    def publish_pose(self, result: Dict):
        """Publish current pose estimate."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "odom"

        # Position
        pose_msg.pose.position.x = float(result["position"][0])
        pose_msg.pose.position.y = float(result["position"][1])
        pose_msg.pose.position.z = float(result["position"][2])

        # Convert rotation matrix to quaternion
        R = result["orientation"]
        quat = self.rotation_matrix_to_quaternion(R)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w])

    def publish_images(self, result: Dict):
        """Publish visualization images."""
        try:
            # Debug image (annotated camera view)
            debug_msg = self.bridge.cv2_to_imgmsg(result["annotated_frame"], "bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_pub.publish(debug_msg)

            # Map image
            map_msg = self.bridge.cv2_to_imgmsg(result["map_image"], "bgr8")
            map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(map_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish images: {e}")

    def publish_status(self, result: Dict):
        """Publish VSLAM status as JSON."""
        status = {
            "frame_count": self.vslam.frame_count,
            "keyframes": self.vslam.stats["keyframes"],
            "loop_closures": self.vslam.stats["loop_closures"],
            "relocalizations": self.vslam.stats["relocalizations"],
            "tracking_quality": self.vslam.stats["tracking_quality"],
            "position": {
                "x": float(result["position"][0]),
                "y": float(result["position"][1]),
                "z": float(result["position"][2]),
            },
            "is_lost": result["is_lost"],
            "is_keyframe": result["is_keyframe"],
            "has_loop_closure": result["loop_closure"] is not None,
            "fps": self.fps,
        }

        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        """Clean up on shutdown."""
        cv2.destroyAllWindows()

        # Print final statistics
        self.get_logger().info("=" * 60)
        self.get_logger().info("FINAL STATISTICS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Total frames: {self.vslam.stats['total_frames']}")
        self.get_logger().info(f"Keyframes: {self.vslam.stats['keyframes']}")
        self.get_logger().info(f"Loop closures: {self.vslam.stats['loop_closures']}")
        self.get_logger().info(f"Relocalizations: {self.vslam.stats['relocalizations']}")
        self.get_logger().info("=" * 60)

        super().destroy_node()


# =============================================================================
# MAIN
# =============================================================================


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = VSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
