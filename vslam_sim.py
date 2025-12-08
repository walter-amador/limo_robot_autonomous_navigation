"""
Visual SLAM (Simultaneous Localization and Mapping) for Monocular Camera
=========================================================================

This implementation provides a 2D visual odometry and mapping system for the
AgileX LIMO robot using a single (monocular) camera.

METHODOLOGY:
------------
1. FEATURE-BASED VISUAL ODOMETRY:
   - Extract ORB (Oriented FAST and Rotated BRIEF) features from each frame
   - Match features between consecutive frames using FLANN-based matcher
   - Estimate camera motion using Essential Matrix decomposition
   - Dead reckoning: accumulate transformations to track robot position

2. LOOP CLOSURE DETECTION (Kidnapped Robot Problem):
   - Build a visual vocabulary using K-Means clustering (Bag of Visual Words)
   - Represent each keyframe as a TF-IDF weighted histogram of visual words
   - Use inverted index for O(1) candidate retrieval (not O(N) linear search)
   - Relocalize by finding the best matching keyframe in the vocabulary

3. 2D MAP BUILDING:
   - Project robot trajectory onto a 2D occupancy-style map
   - Store visited locations and their visual signatures
   - Visualize the path the robot has traveled

4. PLACE RECOGNITION:
   - Use BoVW similarity to recognize previously visited places in O(MxV) time
   - Correct accumulated drift when loop closure is detected using linear interpolation

KEY COMPONENTS:
- FeatureExtractor: ORB feature detection and description
- FeatureMatcher: FLANN-based feature matching with ratio test
- VisualVocabulary: Bag of Visual Words with K-Means clustering and TF-IDF
- VisualOdometry: Frame-to-frame motion estimation
- KeyframeDatabase: Store and retrieve keyframes using BoVW inverted index
- Map2D: 2D visualization of robot trajectory
- LoopClosureDetector: Detect when robot revisits a location
- VSLAMSystem: Main system integrating all components

COMPLEXITY ANALYSIS:
- Brute-force (old): O(N × M²) where N=keyframes, M=features per frame
- BoVW (new): O(M × V + C) where V=vocabulary_size (~1000), C=candidates (~10)
- This enables real-time operation even with thousands of keyframes

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

    # Camera Intrinsics (default values, should be calibrated)
    # These are approximate values for a typical robot camera
    focal_length: float = 500.0  # Focal length in pixels
    principal_point: Tuple[float, float] = (320.0, 240.0)  # Principal point (cx, cy)

    # Map Visualization
    map_scale: float = 50.0  # Pixels per meter in the 2D map
    map_size: Tuple[int, int] = (800, 800)  # Map canvas size

    # Simulation
    simulated_scale: float = 0.1  # Scale factor for simulated motion


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
            table_number=6,  # 6-12 is good
            key_size=12,  # 12-20 is good
            multi_probe_level=1,  # 1-2 is good
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
            # KNN match with k=2 for ratio test
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

    Instead of O(N × M²) brute-force matching against all keyframes,
    this approach:
    1. Builds a visual vocabulary (codebook) using K-Means clustering
    2. Represents each keyframe as a histogram of visual words (BoW vector)
    3. Uses TF-IDF weighting for better discrimination
    4. Retrieves candidates in O(K) where K << N using inverted index

    Complexity: O(M × V) for query, where V = vocabulary size (typically 1000-10000)
    vs O(N × M²) for brute force where N = number of keyframes
    """

    def __init__(self, vocabulary_size: int = 1000, min_samples_for_training: int = 50):
        """
        Args:
            vocabulary_size: Number of visual words (K-Means clusters)
            min_samples_for_training: Minimum descriptor samples before training vocabulary
        """
        self.vocabulary_size = vocabulary_size
        self.min_samples_for_training = min_samples_for_training

        # Visual vocabulary (cluster centers)
        self.vocabulary: Optional[np.ndarray] = None
        self.is_trained = False

        # Inverted index: word_id -> list of (keyframe_id, tf) tuples
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        # Document frequency: word_id -> number of keyframes containing this word
        self.document_frequency: Dict[int, int] = defaultdict(int)

        # BoW vectors for each keyframe (for similarity comparison)
        self.bow_vectors: Dict[int, np.ndarray] = {}

        # Total number of keyframes indexed
        self.num_keyframes = 0

        # Accumulated descriptors for training (before vocabulary is built)
        self.training_descriptors: List[np.ndarray] = []

    def _convert_binary_to_float(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert ORB binary descriptors to float for K-Means."""
        # ORB descriptors are 32 bytes (256 bits) per descriptor
        # Convert each byte to 8 float values (0 or 1)
        if descriptors is None or len(descriptors) == 0:
            return np.array([])

        # Unpack bits: each descriptor becomes 256 float values
        float_desc = np.unpackbits(descriptors, axis=1).astype(np.float32)
        return float_desc

    def _convert_float_to_binary(self, float_desc: np.ndarray) -> np.ndarray:
        """Convert float descriptors back to binary (for vocabulary centers)."""
        # Threshold at 0.5 and pack bits
        binary = (float_desc > 0.5).astype(np.uint8)
        packed = np.packbits(binary, axis=1)
        return packed

    def add_training_sample(self, descriptors: np.ndarray):
        """
        Add descriptors to training pool (used before vocabulary is trained).

        Args:
            descriptors: ORB descriptors from a frame
        """
        if descriptors is not None and len(descriptors) > 0:
            self.training_descriptors.append(descriptors.copy())

    def train_vocabulary(self, descriptors_list: List[np.ndarray] = None):
        """
        Train the visual vocabulary using K-Means clustering.

        Args:
            descriptors_list: Optional list of descriptor arrays. If None, uses accumulated samples.
        """
        if descriptors_list is None:
            descriptors_list = self.training_descriptors

        if len(descriptors_list) == 0:
            print("BoVW: No training samples available")
            return False

        # Concatenate all descriptors
        all_descriptors = np.vstack(descriptors_list)

        if len(all_descriptors) < self.vocabulary_size:
            print(
                f"BoVW: Not enough descriptors ({len(all_descriptors)}) for vocabulary size ({self.vocabulary_size})"
            )
            return False

        # Convert binary to float for K-Means
        float_descriptors = self._convert_binary_to_float(all_descriptors)

        # Aggressively subsample for fast training (10k samples is plenty for vocabulary)
        max_samples = 10000
        if len(float_descriptors) > max_samples:
            print(
                f"BoVW: Subsampling {len(float_descriptors)} -> {max_samples} descriptors for training..."
            )
            indices = np.random.choice(
                len(float_descriptors), max_samples, replace=False
            )
            float_descriptors = float_descriptors[indices]

        print(f"BoVW: Training vocabulary with {len(float_descriptors)} descriptors...")

        try:
            # Run K-Means clustering with faster settings
            # Use 'random' init (faster than '++') and fewer iterations
            centroids, labels = kmeans2(
                float_descriptors,
                self.vocabulary_size,
                minit="random",  # Random initialization (faster)
                iter=10,  # Fewer iterations
                missing="warn",
            )

            self.vocabulary = centroids
            self.is_trained = True

            # Clear training samples to free memory
            self.training_descriptors = []

            print(f"BoVW: Vocabulary trained with {self.vocabulary_size} visual words")
            return True

        except Exception as e:
            print(f"BoVW: Training failed - {e}")
            return False

    def _quantize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Quantize descriptors to visual word IDs.

        Args:
            descriptors: ORB descriptors (N x 32 bytes)

        Returns:
            word_ids: Array of visual word IDs for each descriptor
        """
        if not self.is_trained or descriptors is None or len(descriptors) == 0:
            return np.array([])

        # Convert to float
        float_desc = self._convert_binary_to_float(descriptors)

        # Find nearest cluster center for each descriptor
        word_ids, _ = vq(float_desc, self.vocabulary)

        return word_ids

    def compute_bow_vector(
        self, descriptors: np.ndarray, use_tfidf: bool = True
    ) -> np.ndarray:
        """
        Compute Bag of Words histogram for a set of descriptors.

        Args:
            descriptors: ORB descriptors
            use_tfidf: Whether to apply TF-IDF weighting

        Returns:
            bow_vector: Normalized histogram of visual words
        """
        if not self.is_trained:
            return np.zeros(self.vocabulary_size)

        word_ids = self._quantize_descriptors(descriptors)

        if len(word_ids) == 0:
            return np.zeros(self.vocabulary_size)

        # Compute term frequency (TF)
        bow = np.bincount(word_ids, minlength=self.vocabulary_size).astype(np.float64)

        # Normalize TF
        if bow.sum() > 0:
            bow = bow / bow.sum()

        # Apply TF-IDF weighting
        if use_tfidf and self.num_keyframes > 0:
            for word_id in range(self.vocabulary_size):
                df = self.document_frequency.get(word_id, 0)
                if df > 0:
                    # IDF = log(N / df)
                    idf = np.log(self.num_keyframes / df)
                    bow[word_id] *= idf

        # L2 normalize
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm

        return bow

    def add_keyframe(self, keyframe_id: int, descriptors: np.ndarray):
        """
        Add a keyframe to the database with its BoW representation.

        Args:
            keyframe_id: Unique keyframe identifier
            descriptors: ORB descriptors for this keyframe
        """
        if not self.is_trained:
            # Accumulate for training
            self.add_training_sample(descriptors)
            return

        if descriptors is None or len(descriptors) == 0:
            return

        # Quantize descriptors to visual words
        word_ids = self._quantize_descriptors(descriptors)

        if len(word_ids) == 0:
            return

        # Compute term frequencies
        word_counts = np.bincount(word_ids, minlength=self.vocabulary_size)
        total_words = len(word_ids)

        # Update inverted index and document frequency
        unique_words = np.where(word_counts > 0)[0]
        for word_id in unique_words:
            tf = word_counts[word_id] / total_words
            self.inverted_index[word_id].append((keyframe_id, tf))
            self.document_frequency[word_id] += 1

        # Store BoW vector for this keyframe
        self.bow_vectors[keyframe_id] = self.compute_bow_vector(
            descriptors, use_tfidf=False
        )

        self.num_keyframes += 1

    def query(
        self, descriptors: np.ndarray, top_k: int = 10, exclude_recent: int = 0
    ) -> List[Tuple[int, float]]:
        """
        Query the database for similar keyframes using BoVW.

        This is O(M × V + C) where:
        - M = number of query descriptors
        - V = vocabulary size
        - C = number of candidate keyframes (typically small due to inverted index)

        Args:
            descriptors: Query descriptors
            top_k: Number of top matches to return
            exclude_recent: Number of recent keyframes to exclude

        Returns:
            matches: List of (keyframe_id, similarity_score) tuples, sorted by score
        """
        if not self.is_trained or descriptors is None or len(descriptors) == 0:
            return []

        # Compute query BoW vector
        query_bow = self.compute_bow_vector(descriptors)

        if np.linalg.norm(query_bow) == 0:
            return []

        # Find candidate keyframes using inverted index
        # Only consider keyframes that share at least one visual word with query
        word_ids = self._quantize_descriptors(descriptors)
        unique_words = np.unique(word_ids)

        candidate_scores: Dict[int, float] = defaultdict(float)

        # Accumulate scores from inverted index
        for word_id in unique_words:
            if word_id not in self.inverted_index:
                continue

            # IDF weight for this word
            df = self.document_frequency.get(word_id, 1)
            idf = np.log(max(self.num_keyframes, 1) / df) if df > 0 else 0

            # Query term frequency
            query_tf = np.sum(word_ids == word_id) / len(word_ids)

            for kf_id, kf_tf in self.inverted_index[word_id]:
                # Skip recent keyframes
                if self.num_keyframes - kf_id <= exclude_recent:
                    continue

                # Accumulate TF-IDF score contribution
                candidate_scores[kf_id] += query_tf * kf_tf * (idf**2)

        if len(candidate_scores) == 0:
            return []

        # Refine top candidates with full BoW similarity
        # Sort by accumulated score and take top candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        top_candidates = sorted_candidates[: min(top_k * 3, len(sorted_candidates))]

        # Compute precise cosine similarity for top candidates
        results = []
        for kf_id, _ in top_candidates:
            if kf_id in self.bow_vectors:
                kf_bow = self.bow_vectors[kf_id]
                # Cosine similarity
                similarity = np.dot(query_bow, kf_bow)
                results.append((kf_id, similarity))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def get_statistics(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            "is_trained": self.is_trained,
            "vocabulary_size": self.vocabulary_size,
            "num_keyframes": self.num_keyframes,
            "num_training_samples": len(self.training_descriptors),
            "index_size": sum(len(v) for v in self.inverted_index.values()),
        }


# =============================================================================
# VISUAL ODOMETRY
# =============================================================================


class VisualOdometry:
    """
    Estimates camera motion between consecutive frames.

    Process:
    1. Extract features from current frame
    2. Match with previous frame
    3. Use matched points to estimate Essential Matrix
    4. Decompose Essential Matrix to get rotation and translation
    5. Scale translation (monocular ambiguity - we use unit scale)

    Note: Monocular VO cannot recover absolute scale. In real application,
    you'd fuse with wheel odometry or IMU for scale estimation.
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

        # Tracking statistics
        self.num_matches = 0
        self.tracking_quality = 1.0
        self.frames_without_matches = 0

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a new frame and estimate motion.

        Args:
            image: Input BGR image

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

            # Still update previous frame for next iteration
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

        # Apply simulated scale (in real application, fuse with wheel odometry)
        scale = self.config.simulated_scale

        # Update pose using dead reckoning
        # The translation is in camera frame, we need to transform to world frame
        t_world = self.orientation @ (t.flatten() * scale)
        self.position += t_world
        self.orientation = R @ self.orientation

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_image = image.copy()

        return self.position.copy(), self.orientation.copy(), len(matches)

    def is_lost(self) -> bool:
        """Check if visual tracking is lost."""
        return self.frames_without_matches >= self.config.lost_tracking_frames

    def reset_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Reset pose after relocalization."""
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
    thumbnail: np.ndarray = None  # Small version of image for display

    def __post_init__(self):
        # Convert keypoints to serializable format for saving
        self.keypoints_data = [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in self.keypoints
        ]


class KeyframeDatabase:
    """
    Database of keyframes for loop closure and relocalization.

    This implements an efficient Bag of Visual Words (BoVW) approach:
    - Build a visual vocabulary using K-Means clustering on ORB descriptors
    - Represent each keyframe as a histogram of visual words
    - Use inverted index for O(1) candidate retrieval instead of O(N) linear search
    - Apply TF-IDF weighting for better discrimination

    Complexity: O(M × V + C) for query vs O(N × M²) for brute force
    where M=descriptors, V=vocabulary_size, C=candidates, N=keyframes
    """

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.keyframes: List[Keyframe] = []
        self.matcher = FeatureMatcher(config)
        self.next_id = 0

        # Initialize Bag of Visual Words vocabulary
        self.vocabulary = VisualVocabulary(
            vocabulary_size=500,  # Number of visual words (smaller = faster)
            min_samples_for_training=30,  # Train after collecting enough samples
        )

        # Track when vocabulary was last trained
        self.vocab_trained_at_keyframe = -1

    def should_add_keyframe(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> bool:
        """Determine if current pose warrants a new keyframe."""
        if len(self.keyframes) == 0:
            return True

        last_kf = self.keyframes[-1]

        # Check distance
        distance = np.linalg.norm(position - last_kf.position)
        if distance >= self.config.keyframe_min_distance:
            return True

        # Check rotation (using trace of rotation difference)
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
        """Add a new keyframe to the database."""

        # Create thumbnail for visualization
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

        # Add to BoVW vocabulary/index
        if descriptors is not None:
            # If vocabulary not trained yet, add to training samples
            if not self.vocabulary.is_trained:
                self.vocabulary.add_training_sample(descriptors)

                # Try to train vocabulary when we have enough samples
                if (
                    len(self.vocabulary.training_descriptors)
                    >= self.vocabulary.min_samples_for_training
                ):
                    if self.vocabulary.train_vocabulary():
                        self.vocab_trained_at_keyframe = self.next_id
                        # Re-index all existing keyframes
                        self._reindex_all_keyframes()
            else:
                # Add keyframe to vocabulary index
                self.vocabulary.add_keyframe(self.next_id, descriptors)

        self.next_id += 1
        return kf.id

    def _reindex_all_keyframes(self):
        """Re-index all existing keyframes after vocabulary training."""
        print(f"BoVW: Re-indexing {len(self.keyframes)} existing keyframes...")
        for kf in self.keyframes:
            if kf.descriptors is not None:
                self.vocabulary.add_keyframe(kf.id, kf.descriptors)
        print(f"BoVW: Re-indexing complete")

    def find_best_match(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        """
        Find the keyframe that best matches the given descriptors.

        Uses Bag of Visual Words (BoVW) for efficient retrieval:
        1. Query the BoVW index for candidate keyframes (O(M × V))
        2. Verify top candidates with geometric consistency (optional)

        Falls back to brute-force only if vocabulary is not trained yet.

        Args:
            descriptors: Current frame descriptors
            exclude_recent: Number of recent keyframes to exclude (for loop closure)

        Returns:
            best_keyframe: Best matching keyframe or None
            best_score: Normalized match score (0-1)
            num_matches: Number of feature matches (or BoW similarity × 100)
        """
        if descriptors is None or len(self.keyframes) == 0:
            return None, 0.0, 0

        # Use BoVW if vocabulary is trained
        if self.vocabulary.is_trained:
            return self._find_best_match_bovw(descriptors, exclude_recent)
        else:
            # Fallback to brute-force for early keyframes (before vocabulary training)
            return self._find_best_match_bruteforce(descriptors, exclude_recent)

    def _find_best_match_bovw(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        """
        Efficient BoVW-based matching.

        Complexity: O(M × V + C) where M=descriptors, V=vocab_size, C=candidates
        """
        # Query BoVW index for top candidates
        candidates = self.vocabulary.query(
            descriptors, top_k=5, exclude_recent=exclude_recent  # Get top 5 candidates
        )

        if len(candidates) == 0:
            return None, 0.0, 0

        # Get best candidate
        best_kf_id, best_score = candidates[0]

        # Retrieve the keyframe
        best_keyframe = None
        for kf in self.keyframes:
            if kf.id == best_kf_id:
                best_keyframe = kf
                break

        if best_keyframe is None:
            return None, 0.0, 0

        # Optionally: Verify with geometric consistency (feature matching)
        # This adds robustness but increases computation
        # For now, we trust the BoVW score

        # Convert similarity to pseudo match count for compatibility
        pseudo_matches = int(best_score * 100)

        return best_keyframe, best_score, pseudo_matches

    def _find_best_match_bruteforce(
        self, descriptors: np.ndarray, exclude_recent: int = 0
    ) -> Tuple[Optional[Keyframe], float, int]:
        """
        Brute-force matching (fallback before vocabulary is trained).

        Complexity: O(N × M²) - only used for first few keyframes
        """
        best_keyframe = None
        best_score = 0.0
        best_matches = 0

        # Search through keyframes (excluding recent ones if specified)
        search_range = len(self.keyframes) - exclude_recent

        for i in range(search_range):
            kf = self.keyframes[i]
            if kf.descriptors is None:
                continue

            matches = self.matcher.match(descriptors, kf.descriptors)
            num_matches = len(matches)

            # Normalize score by number of possible matches
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
        """Get array of all keyframe positions."""
        if len(self.keyframes) == 0:
            return np.array([])
        return np.array([kf.position for kf in self.keyframes])

    def save(self, filepath: str):
        """Save database to file."""
        # Prepare data for serialization
        data = {
            "keyframes": [
                (
                    kf.id,
                    kf.timestamp,
                    kf.position,
                    kf.orientation,
                    kf.descriptors,
                    kf.keypoints_data,
                )
                for kf in self.keyframes
            ],
            "next_id": self.next_id,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load database from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.keyframes = []
        self.next_id = data["next_id"]

        for kf_data in data["keyframes"]:
            kf_id, timestamp, position, orientation, descriptors, kp_data = kf_data
            # Reconstruct keypoints
            keypoints = [
                cv2.KeyPoint(
                    x=pt[0],
                    y=pt[1],
                    size=size,
                    angle=angle,
                    response=response,
                    octave=octave,
                    class_id=class_id,
                )
                for pt, size, angle, response, octave, class_id in kp_data
            ]

            kf = Keyframe(
                id=kf_id,
                timestamp=timestamp,
                position=position,
                orientation=orientation,
                descriptors=descriptors,
                keypoints=keypoints,
            )
            self.keyframes.append(kf)


# =============================================================================
# LOOP CLOSURE DETECTOR
# =============================================================================


class LoopClosureDetector:
    """
    Detects when the robot returns to a previously visited location.

    This is crucial for:
    1. Correcting accumulated drift in dead reckoning
    2. Detecting the "kidnapped robot" scenario

    Method:
    - Compare current frame features with keyframe database
    - If high similarity with a non-recent keyframe = loop closure
    - Use geometric verification to confirm
    """

    def __init__(self, config: VSLAMConfig, keyframe_db: KeyframeDatabase):
        self.config = config
        self.keyframe_db = keyframe_db
        self.matcher = FeatureMatcher(config)

        # Loop closure history
        self.detected_loops: List[Tuple[int, int, float]] = (
            []
        )  # (current_kf, matched_kf, score)

    def check_loop_closure(
        self, descriptors: np.ndarray, current_kf_id: int
    ) -> Tuple[bool, Optional[Keyframe], float]:
        """
        Check if current frame closes a loop with a previous keyframe.

        Args:
            descriptors: Current frame descriptors
            current_kf_id: Current keyframe ID

        Returns:
            is_loop: True if loop closure detected
            matched_keyframe: The keyframe that was matched
            confidence: Match confidence score
        """
        # Find best matching keyframe, excluding recent ones
        best_kf, score, num_matches = self.keyframe_db.find_best_match(
            descriptors, exclude_recent=self.config.loop_closure_min_frames
        )

        if best_kf is None:
            return False, None, 0.0

        # Check if score exceeds threshold
        if score >= self.config.loop_closure_threshold:
            # Additional geometric verification could go here
            # (e.g., check if matched points form a consistent transformation)

            self.detected_loops.append((current_kf_id, best_kf.id, score))
            return True, best_kf, score

        return False, None, score


# =============================================================================
# 2D MAP VISUALIZATION
# =============================================================================


class Map2D:
    """
    2D visualization of robot trajectory and mapping.

    Creates a top-down view showing:
    - Robot trajectory (path taken)
    - Keyframe locations
    - Current robot position
    - Loop closures (when detected)
    """

    def __init__(self, config: VSLAMConfig):
        self.config = config
        self.width, self.height = config.map_size
        self.scale = config.map_scale  # pixels per meter

        # Map center (robot starts here)
        self.origin = np.array([self.width // 2, self.height // 2])

        # Trajectory history
        self.trajectory: List[np.ndarray] = []
        self.keyframe_positions: List[np.ndarray] = []
        self.loop_closures: List[Tuple[np.ndarray, np.ndarray]] = (
            []
        )  # pairs of positions

    def world_to_map(self, position: np.ndarray) -> Tuple[int, int]:
        """
        Convert world position to map pixel coordinates.

        Camera coordinate system:
        - X: right
        - Y: down
        - Z: forward (into the scene)

        Map coordinate system:
        - map_x: right (camera X)
        - map_y: up on screen = forward movement (camera Z)
        """
        # X maps to horizontal axis
        map_x = int(self.origin[0] + position[0] * self.scale)
        # Z (forward) maps to vertical axis, negated so forward = up on screen
        map_y = int(self.origin[1] - position[2] * self.scale)
        return map_x, map_y

    def add_position(self, position: np.ndarray):
        """Add a position to the trajectory."""
        self.trajectory.append(position.copy())

    def add_keyframe(self, position: np.ndarray):
        """Mark a keyframe position."""
        self.keyframe_positions.append(position.copy())

    def add_loop_closure(self, pos1: np.ndarray, pos2: np.ndarray):
        """Add a loop closure connection."""
        self.loop_closures.append((pos1.copy(), pos2.copy()))

    def refresh_trajectory(self, keyframe_positions: List[np.ndarray]):
        """
        Refresh the trajectory and keyframe positions after loop closure correction.

        Args:
            keyframe_positions: Updated list of keyframe positions from the database
        """
        # Update keyframe positions
        self.keyframe_positions = [pos.copy() for pos in keyframe_positions]

        # Rebuild trajectory from keyframe positions
        # This gives us the corrected path
        self.trajectory = [pos.copy() for pos in keyframe_positions]

    def render(
        self, current_position: np.ndarray, current_orientation: np.ndarray
    ) -> np.ndarray:
        """
        Render the current map.

        Args:
            current_position: Current robot position
            current_orientation: Current robot orientation (3x3)

        Returns:
            map_image: BGR image of the map
        """
        # Create blank map (dark background)
        map_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        map_img[:] = (40, 40, 40)  # Dark gray background

        # Draw grid
        grid_spacing = int(1.0 * self.scale)  # 1 meter grid
        for x in range(0, self.width, grid_spacing):
            cv2.line(map_img, (x, 0), (x, self.height), (60, 60, 60), 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(map_img, (0, y), (self.width, y), (60, 60, 60), 1)

        # Draw origin marker
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
                # Check bounds
                if (
                    0 <= pt1[0] < self.width
                    and 0 <= pt1[1] < self.height
                    and 0 <= pt2[0] < self.width
                    and 0 <= pt2[1] < self.height
                ):
                    cv2.line(map_img, pt1, pt2, (255, 200, 0), 2)  # Cyan trajectory

        # Draw keyframes
        for kf_pos in self.keyframe_positions:
            pt = self.world_to_map(kf_pos)
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height:
                cv2.circle(map_img, pt, 5, (0, 255, 0), -1)  # Green keyframes

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
                cv2.line(map_img, pt1, pt2, (255, 0, 255), 2)  # Magenta loop closures

        # Draw current robot position and heading
        robot_pt = self.world_to_map(current_position)
        if 0 <= robot_pt[0] < self.width and 0 <= robot_pt[1] < self.height:
            # Robot body
            cv2.circle(map_img, robot_pt, 8, (0, 0, 255), -1)  # Red robot

            # Robot heading (arrow)
            # Get forward direction from orientation matrix (Z-axis in camera frame)
            # Z-axis (column 2) points forward in camera coordinates
            forward = current_orientation[:, 2]  # Third column = forward direction
            # Map forward direction to screen: X->right, Z->up (negated)
            arrow_length = 30
            heading_end = (
                int(
                    robot_pt[0] + forward[0] * arrow_length
                ),  # X component -> horizontal
                int(
                    robot_pt[1] - forward[2] * arrow_length
                ),  # Z component -> vertical (negated for screen)
            )
            cv2.arrowedLine(
                map_img, robot_pt, heading_end, (0, 0, 255), 2, tipLength=0.3
            )

        # Add legend
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

        # Scale indicator
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
    """
    Main Visual SLAM system integrating all components.

    Workflow:
    1. Process each frame through visual odometry
    2. Detect if we need a new keyframe
    3. Check for loop closure with previous keyframes
    4. Handle "kidnapped robot" through relocalization
    5. Update and visualize the map
    """

    def __init__(self, config: VSLAMConfig = None):
        self.config = config or VSLAMConfig()

        # Initialize components
        self.visual_odometry = VisualOdometry(self.config)
        self.keyframe_db = KeyframeDatabase(self.config)
        self.loop_detector = LoopClosureDetector(self.config, self.keyframe_db)
        self.map_2d = Map2D(self.config)
        self.feature_extractor = FeatureExtractor(self.config)

        # State
        self.frame_count = 0
        self.current_keyframe_id = -1
        self.is_relocalized = False
        self.relocalization_source = None

        # Statistics
        self.stats = {
            "total_frames": 0,
            "keyframes": 0,
            "loop_closures": 0,
            "relocalizations": 0,
            "tracking_quality": 1.0,
        }

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the VSLAM pipeline.

        Args:
            frame: Input BGR image

        Returns:
            result: Dictionary containing:
                - position: Current 3D position
                - orientation: Current 3x3 rotation matrix
                - is_keyframe: True if this frame became a keyframe
                - loop_closure: Info about detected loop closure (if any)
                - is_lost: True if tracking is lost
                - map_image: Rendered 2D map
                - annotated_frame: Frame with VSLAM visualization
        """
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

        # Step 1: Visual Odometry
        position, orientation, num_matches = self.visual_odometry.process_frame(frame)
        result["position"] = position
        result["orientation"] = orientation
        self.stats["tracking_quality"] = self.visual_odometry.tracking_quality

        # Add position to trajectory
        self.map_2d.add_position(position)

        # Step 2: Check if tracking is lost (kidnapped robot scenario)
        if self.visual_odometry.is_lost():
            result["is_lost"] = True

            # Attempt relocalization
            keypoints, descriptors = self.feature_extractor.extract(frame)
            best_kf, score, matches = self.keyframe_db.find_best_match(descriptors)

            if best_kf is not None and score >= self.config.relocalization_threshold:
                # Relocalize to matched keyframe
                self.visual_odometry.reset_pose(best_kf.position, best_kf.orientation)
                result["position"] = best_kf.position.copy()
                result["orientation"] = best_kf.orientation.copy()
                result["relocalized"] = True
                self.is_relocalized = True
                self.relocalization_source = best_kf.id
                self.stats["relocalizations"] += 1
                print(f"RELOCALIZED to keyframe {best_kf.id} (score: {score:.2f})")

        # Step 3: Keyframe selection
        keypoints, descriptors = self.feature_extractor.extract(frame)

        if self.keyframe_db.should_add_keyframe(position, orientation):
            kf_id = self.keyframe_db.add_keyframe(
                position, orientation, descriptors, keypoints, frame
            )
            self.current_keyframe_id = kf_id
            result["is_keyframe"] = True
            self.stats["keyframes"] = len(self.keyframe_db.keyframes)
            self.map_2d.add_keyframe(position)

            # Step 4: Check for loop closure
            if len(self.keyframe_db.keyframes) > self.config.loop_closure_min_frames:
                is_loop, matched_kf, confidence = self.loop_detector.check_loop_closure(
                    descriptors, kf_id
                )

                if is_loop:
                    # Store pre-correction position error
                    position_error_before = np.linalg.norm(
                        position - matched_kf.position
                    )

                    result["loop_closure"] = {
                        "matched_keyframe": matched_kf.id,
                        "confidence": confidence,
                        "position_error": position_error_before,
                    }
                    self.stats["loop_closures"] += 1

                    # Add loop closure visualization (before correction)
                    self.map_2d.add_loop_closure(position, matched_kf.position)

                    print(
                        f"LOOP CLOSURE detected with keyframe {matched_kf.id} "
                        f"(confidence: {confidence:.2f}, drift: {position_error_before:.3f}m)"
                    )

                    # Perform linear drift correction
                    corrected_pos = self.perform_linear_correction(position, matched_kf)

                    # Update visual odometry position to corrected location
                    self.visual_odometry.position = corrected_pos.copy()

                    # Update the current keyframe's position as well
                    self.keyframe_db.keyframes[kf_id].position = corrected_pos.copy()

                    # Update result with corrected position
                    position = corrected_pos
                    result["position"] = position

                    # Refresh Map2D trajectory with corrected keyframe positions
                    corrected_kf_positions = [
                        kf.position for kf in self.keyframe_db.keyframes
                    ]
                    self.map_2d.refresh_trajectory(corrected_kf_positions)

                    print(
                        f"  Drift corrected. New position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
                    )

        # Step 5: Render visualizations
        result["map_image"] = self.map_2d.render(position, orientation)
        result["annotated_frame"] = self._annotate_frame(frame, result, num_matches)

        return result

    def _annotate_frame(
        self, frame: np.ndarray, result: Dict, num_matches: int
    ) -> np.ndarray:
        """Add VSLAM information overlay to frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Draw features (optional, can be toggled)
        keypoints, _ = self.feature_extractor.extract(frame)
        for kp in keypoints[:100]:  # Draw first 100 features
            pt = tuple(map(int, kp.pt))
            cv2.circle(annotated, pt, 3, (0, 255, 255), 1)

        # Status panel background
        cv2.rectangle(annotated, (5, 5), (250, 160), (0, 0, 0), -1)
        cv2.rectangle(annotated, (5, 5), (250, 160), (255, 255, 255), 1)

        # Position info
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

        # Tracking info
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

        # Tracking quality bar
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

        # Status indicators
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
        """
        Perform linear drift correction across keyframes when loop closure is detected.

        This implements a simple but effective correction strategy:
        1. Calculate the drift vector between current position and matched keyframe position
        2. Distribute this error linearly across all keyframes between the matched one and current
        3. Update all affected keyframe positions

        Args:
            current_pos: Current estimated position (before correction)
            matched_kf: The keyframe that was matched in loop closure

        Returns:
            corrected_pos: The corrected current position
        """
        matched_kf_id = matched_kf.id
        current_kf_id = self.current_keyframe_id

        # Ensure we have a valid range to correct
        if current_kf_id <= matched_kf_id:
            return current_pos.copy()

        # Calculate drift vector: the error accumulated since the matched keyframe
        # Drift = where we think we are - where we should be (matched keyframe position)
        drift_vector = current_pos - matched_kf.position

        # Number of keyframes to distribute the correction across
        num_keyframes_to_correct = current_kf_id - matched_kf_id

        if num_keyframes_to_correct <= 0:
            return current_pos.copy()

        print(
            f"  Correcting drift: {np.linalg.norm(drift_vector):.3f}m across {num_keyframes_to_correct} keyframes"
        )

        # Apply linear correction to each keyframe between matched and current
        # The correction is proportional to how far along the path the keyframe is
        for i, kf in enumerate(self.keyframe_db.keyframes):
            if kf.id <= matched_kf_id:
                # Keyframes at or before the matched one don't need correction
                continue
            if kf.id > current_kf_id:
                # Keyframes after current shouldn't exist yet, but skip if they do
                continue

            # Calculate interpolation factor (0 at matched_kf, 1 at current_kf)
            # This determines how much of the drift to apply to this keyframe
            alpha = (kf.id - matched_kf_id) / num_keyframes_to_correct

            # Apply proportional correction (subtract drift proportionally)
            # Keyframes closer to matched_kf get less correction
            # Keyframes closer to current get more correction
            correction = drift_vector * alpha
            kf.position = kf.position - correction

        # The corrected current position should now match the matched keyframe
        # (since we're closing a loop, we should be at the same place)
        corrected_pos = current_pos - drift_vector

        return corrected_pos

    def save_map(self, filepath: str):
        """Save the keyframe database for later use."""
        self.keyframe_db.save(filepath)
        print(f"Map saved to {filepath}")

    def load_map(self, filepath: str):
        """Load a previously saved keyframe database."""
        if os.path.exists(filepath):
            self.keyframe_db.load(filepath)
            print(
                f"Map loaded from {filepath} ({len(self.keyframe_db.keyframes)} keyframes)"
            )
        else:
            print(f"Map file not found: {filepath}")


# =============================================================================
# SIMULATION / DEMO
# =============================================================================


def init_camera(source):
    """Initialize video capture from camera or file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def main():
    """
    Main function demonstrating the VSLAM system.

    Controls:
        ESC - Exit
        S   - Save map
        L   - Load map
        R   - Reset SLAM
        K   - Simulate kidnap (random relocalization test)
    """

    # Configuration
    config = VSLAMConfig(
        n_features=1000,
        match_ratio=0.75,
        min_matches=30,
        keyframe_min_distance=0.2,
        keyframe_min_rotation=0.1,
        loop_closure_threshold=0.5,
        simulated_scale=0.05,  # Smaller scale for video simulation
        map_scale=100.0,  # Larger scale for better visualization
    )

    # Initialize VSLAM system
    vslam = VSLAMSystem(config)

    # Initialize video source
    # Use recorded video for simulation
    # video_source = "rsrc/camera_recording_20251128_113309.mp4"
    video_source = 2  # Uncomment for live camera

    try:
        cap = init_camera(video_source)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Available video files in rsrc/:")
        print("  - camera_recording_20251128_113309.mp4")
        return

    # Map save path
    map_save_path = "vslam_map.pkl"

    # FPS tracking
    prev_time = time.time()
    fps = 0

    print("=" * 60)
    print("VISUAL SLAM SIMULATION")
    print("=" * 60)
    print("\nControls:")
    print("  ESC - Exit")
    print("  S   - Save map")
    print("  L   - Load map")
    print("  R   - Reset SLAM")
    print("  K   - Simulate kidnap (test relocalization)")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video - restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Process frame through VSLAM
        result = vslam.process_frame(frame)

        # Add FPS to annotated frame
        cv2.putText(
            result["annotated_frame"],
            f"FPS: {fps:.1f}",
            (540, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Display windows
        cv2.imshow("VSLAM - Camera View", result["annotated_frame"])
        cv2.imshow("VSLAM - 2D Map", result["map_image"])

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("s") or key == ord("S"):
            vslam.save_map(map_save_path)
        elif key == ord("l") or key == ord("L"):
            vslam.load_map(map_save_path)
        elif key == ord("r") or key == ord("R"):
            print("Resetting SLAM...")
            vslam = VSLAMSystem(config)
        elif key == ord("k") or key == ord("K"):
            # Simulate kidnap - mess up the pose
            print("SIMULATING KIDNAP - scrambling pose...")
            vslam.visual_odometry.position = np.array(
                [
                    np.random.uniform(-5, 5),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-5, 5),
                ]
            )
            vslam.visual_odometry.frames_without_matches = config.lost_tracking_frames

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total frames processed: {vslam.stats['total_frames']}")
    print(f"Keyframes created: {vslam.stats['keyframes']}")
    print(f"Loop closures detected: {vslam.stats['loop_closures']}")
    print(f"Relocalizations performed: {vslam.stats['relocalizations']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
