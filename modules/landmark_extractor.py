import numpy as np
import mediapipe as mp
from typing import Optional


def extract_landmarks(image):
    # Initialize MediaPipe Hands in static image mode.
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands_detector:

        # Run detection
        result = hands_detector.process(image)

        # If no hands found, return None
        if not result.multi_hand_landmarks:
            return None

        # If multiple hands were somehow detected, pick the one with highest confidence.
        if len(result.multi_hand_landmarks) > 1:
            best_hand_index = _pick_most_confident_hand(result.multi_handedness)
        else:
            best_hand_index = 0

        hand_landmarks = result.multi_hand_landmarks[best_hand_index]

        # Convert MediaPipe's landmark format to a simple (21, 3) numpy array.
        landmarks_array = _landmarks_to_array(hand_landmarks)

        return landmarks_array


def create_video_extractor(detection_confidence: float = 0.7,
                           tracking_confidence: float = 0.5):
    # Create MediaPipe Hands in tracking mode.
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,   # False = tracking mode (reuses detections)
        max_num_hands=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    )

    def extract_from_frame(frame):
        result = hands_detector.process(frame)

        if not result.multi_hand_landmarks:
            return None

        if len(result.multi_hand_landmarks) > 1:
            best_index = _pick_most_confident_hand(result.multi_handedness)
        else:
            best_index = 0

        hand_landmarks = result.multi_hand_landmarks[best_index]
        return _landmarks_to_array(hand_landmarks)

    def close_detector():
        hands_detector.close()

    # Attach the close method to the extractor function so the caller can clean up
    extract_from_frame.close = close_detector

    return extract_from_frame


def _pick_most_confident_hand(multi_handedness):
    best_index = 0
    best_score = 0.0

    for index, handedness in enumerate(multi_handedness):
        # Each handedness has a .classification list with one entry
        score = handedness.classification[0].score
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def _landmarks_to_array(hand_landmarks):
    landmarks_list = []

    for landmark in hand_landmarks.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks_list, dtype=np.float32)
