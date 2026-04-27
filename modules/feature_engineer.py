import numpy as np
from itertools import combinations
from typing import List
import os
import sys
sys.path.append(os.path.abspath('..'))
import config


def compute_features(landmarks):
    # Normalize position (center at wrist) and scale 
    normalized_landmarks = normalize_landmarks(landmarks)

    # Flatten coordinates to a 1D vector 
    coordinate_features = flatten_coordinates(normalized_landmarks)

    # Compute pairwise distances between fingertips
    fingertip_distance_features = compute_fingertip_distances(normalized_landmarks)

    # Compute angles at the PIP (middle) joint of each finger
    pip_angle_features = compute_pip_angles(normalized_landmarks)

    # Compute finger extension ratios (how straight vs curled each finger is)
    extension_ratio_features = compute_extension_ratios(normalized_landmarks)

    # Concatenate all feature groups into one flat vector
    feature_vector = np.concatenate([
        coordinate_features,
        fingertip_distance_features,
        pip_angle_features,
        extension_ratio_features
    ])

    return feature_vector.astype(np.float32)


def normalize_landmarks(landmarks):
    # Make a copy so we don't modify the original array
    normalized = landmarks.copy()

    # Subtract wrist position (landmark 0) from all landmarks
    wrist_position = landmarks[config.WRIST_INDEX]
    normalized = normalized - wrist_position

    # Compute the distance from wrist to middle finger MCP
    middle_mcp_position = normalized[config.MIDDLE_MCP_INDEX]
    palm_scale = np.linalg.norm(middle_mcp_position)  # Euclidean distance

    # Avoid division by zero if somehow both points overlap
    if palm_scale < 1e-6:
        palm_scale = 1e-6

    # Divide all coordinates by the palm scale
    normalized = normalized / palm_scale

    return normalized


def flatten_coordinates(normalized_landmarks):
    if config.DROP_Z:
        # Keep only x and y columns (columns 0 and 1), drop z (column 2)
        xy_only = normalized_landmarks[:, :2]  # shape (21, 2)
        return xy_only.flatten()
    else:
        return normalized_landmarks.flatten()  # shape (63,)


def compute_fingertip_distances(normalized_landmarks):
    fingertip_indices = config.FINGERTIP_INDICES  # [4, 8, 12, 16, 20]
    thumb_tip_index = 4  # Thumb tip is landmark 4

    distances = []

    # Group 1: All pairwise distances among the 5 fingertips
    for index_a, index_b in combinations(fingertip_indices, 2):
        point_a = normalized_landmarks[index_a]
        point_b = normalized_landmarks[index_b]
        distance = np.linalg.norm(point_a - point_b)
        distances.append(distance)

    # Group 2: Distance from each non-thumb fingertip to the thumb tip
    non_thumb_fingertip_indices = [8, 12, 16, 20]  # index, middle, ring, pinky
    thumb_tip_position = normalized_landmarks[thumb_tip_index]

    for fingertip_index in non_thumb_fingertip_indices:
        fingertip_position = normalized_landmarks[fingertip_index]
        distance = np.linalg.norm(fingertip_position - thumb_tip_position)
        distances.append(distance)

    return np.array(distances, dtype=np.float32)


def compute_pip_angles(normalized_landmarks):
    # MCP and DIP index pairs define the bones on either side of each PIP joint
    # Index 0: thumb, 1: index, 2: middle, 3: ring, 4: pinky
    mcp_indices = config.MCP_INDICES   # [1, 5, 9, 13, 17]
    pip_indices = config.PIP_INDICES   # [2, 6, 10, 14, 18]
    dip_indices = config.DIP_INDICES   # [3, 7, 11, 15, 19]

    angles = []

    for mcp_idx, pip_idx, dip_idx in zip(mcp_indices, pip_indices, dip_indices):
        mcp_point = normalized_landmarks[mcp_idx]
        pip_point = normalized_landmarks[pip_idx]
        dip_point = normalized_landmarks[dip_idx]

        # Compute vectors from PIP toward MCP and from PIP toward DIP
        vector_to_mcp = mcp_point - pip_point
        vector_to_dip = dip_point - pip_point

        # Use the dot product formula to get the angle:
        # cos(angle) = (v1 · v2) / (|v1| * |v2|)
        norm_mcp = np.linalg.norm(vector_to_mcp)
        norm_dip = np.linalg.norm(vector_to_dip)

        # Avoid division by zero if two landmarks are in the same spot
        if norm_mcp < 1e-6 or norm_dip < 1e-6:
            angles.append(0.0)
            continue

        cosine_of_angle = np.dot(vector_to_mcp, vector_to_dip) / (norm_mcp * norm_dip)

        # Clamp to [-1, 1] to handle floating point errors before taking arccos
        cosine_of_angle = np.clip(cosine_of_angle, -1.0, 1.0)

        angle_in_radians = np.arccos(cosine_of_angle)
        angles.append(angle_in_radians)

    return np.array(angles, dtype=np.float32)


def compute_extension_ratios(normalized_landmarks):
    ratios = []

    # For each finger: (MCP, PIP, DIP, TIP) indices
    # Thumb: 1, 2, 3, 4
    # Index: 5, 6, 7, 8
    # Middle: 9, 10, 11, 12
    # Ring: 13, 14, 15, 16
    # Pinky: 17, 18, 19, 20
    finger_landmark_groups = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index finger
        [9, 10, 11, 12], # middle finger
        [13, 14, 15, 16],# ring finger
        [17, 18, 19, 20] # pinky
    ]

    for landmark_indices in finger_landmark_groups:
        mcp_idx = landmark_indices[0]
        pip_idx = landmark_indices[1]
        dip_idx = landmark_indices[2]
        tip_idx = landmark_indices[3]

        mcp_point = normalized_landmarks[mcp_idx]
        pip_point = normalized_landmarks[pip_idx]
        dip_point = normalized_landmarks[dip_idx]
        tip_point = normalized_landmarks[tip_idx]

        # Straight-line distance from base (MCP) to tip
        tip_to_mcp_distance = np.linalg.norm(tip_point - mcp_point)

        # Sum of the three bone segment lengths: MCP→PIP + PIP→DIP + DIP→TIP
        segment_mcp_to_pip = np.linalg.norm(pip_point - mcp_point)
        segment_pip_to_dip = np.linalg.norm(dip_point - pip_point)
        segment_dip_to_tip = np.linalg.norm(tip_point - dip_point)
        total_bone_length = segment_mcp_to_pip + segment_pip_to_dip + segment_dip_to_tip

        # Avoid division by zero
        if total_bone_length < 1e-6:
            ratios.append(0.0)
            continue

        extension_ratio = tip_to_mcp_distance / total_bone_length
        ratios.append(extension_ratio)

    return np.array(ratios, dtype=np.float32)


def get_feature_names() -> List[str]:
    feature_names = []

    # Coordinate features
    num_coords = 2 if config.DROP_Z else 3
    coord_labels = ["x", "y"] if config.DROP_Z else ["x", "y", "z"]

    for landmark_idx in range(21):
        for coord in coord_labels:
            feature_names.append(f"landmark_{landmark_idx:02d}_{coord}")

    # Pairwise fingertip distances
    fingertip_indices = config.FINGERTIP_INDICES
    for index_a, index_b in combinations(fingertip_indices, 2):
        feature_names.append(f"dist_tip{index_a}_tip{index_b}")

    # Thumb-to-fingertip distances
    for tip_idx in [8, 12, 16, 20]:
        feature_names.append(f"dist_thumb_to_tip{tip_idx}")

    # PIP joint angles
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for finger_name in finger_names:
        feature_names.append(f"pip_angle_{finger_name}")

    # Extension ratios
    for finger_name in finger_names:
        feature_names.append(f"extension_ratio_{finger_name}")

    return feature_names