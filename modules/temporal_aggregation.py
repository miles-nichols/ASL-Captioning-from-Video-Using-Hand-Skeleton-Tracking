import sys
import os
sys.path.append(os.path.abspath('..'))
from collections import Counter, deque
import config


def aggregate_predictions(predictions):
    if len(predictions) == 0:
        return ""

    print(f"  Aggregating {len(predictions)} frame predictions...")

    # Filter out frames where the model was not confident
    confident_predictions = filter_by_confidence(predictions, config.CONFIDENCE_THRESHOLD)
    print(f"  After confidence filtering: {len(confident_predictions)} frames remain "
          f"(dropped {len(predictions) - len(confident_predictions)} low-confidence frames)")

    if len(confident_predictions) == 0:
        print("  No confident predictions found. Returning empty transcription.")
        return ""

    # Apply sliding window majority vote to smooth out frame-to-frame noise
    smoothed_predictions = apply_sliding_window(confident_predictions, config.WINDOW_SIZE)
    print(f"  Applied sliding window (size={config.WINDOW_SIZE})")

    # Detect stable segments — runs of the same letter long enough to count
    stable_letters = detect_letter_segments(smoothed_predictions, config.MIN_SEGMENT_LENGTH)
    print(f"  Detected {len(stable_letters)} stable letter segments")

    # Join the stable letters into the final transcription string
    transcription = "".join(stable_letters)
    print(f"  Final transcription: '{transcription}'")

    return transcription


def filter_by_confidence(predictions, threshold):
    kept_predictions = []

    for prediction in predictions:
        if prediction["confidence"] >= threshold:
            kept_predictions.append(prediction)

    return kept_predictions


def apply_sliding_window(predictions, window_size):
    smoothed = []
    num_predictions = len(predictions)
    half_window = window_size // 2

    for current_index in range(num_predictions):
        # Define window boundaries (clamped to valid indices)
        window_start = max(0, current_index - half_window)
        window_end = min(num_predictions, current_index + half_window + 1)

        # Collect letters in this window
        window_letters = [predictions[i]["letter"] for i in range(window_start, window_end)]

        # Find the most common letter in the window
        letter_counts = Counter(window_letters)
        most_common_letter, most_common_count = letter_counts.most_common(1)[0]

        # Build the smoothed prediction
        smoothed_prediction = dict(predictions[current_index])  # copy original
        smoothed_prediction["original_letter"] = predictions[current_index]["letter"]
        smoothed_prediction["letter"] = most_common_letter

        smoothed.append(smoothed_prediction)

    return smoothed


def detect_letter_segments(predictions, min_segment_length):
    if len(predictions) == 0:
        return []

    stable_letters = []
    current_letter = predictions[0]["letter"]
    current_segment_length = 1
    last_recorded_letter = None  # Track the last recorded letter to avoid repeats

    for prediction in predictions[1:]:
        letter = prediction["letter"]

        if letter == current_letter:
            # Same letter as before — extend the current segment
            current_segment_length += 1
        else:
            # Letter changed — evaluate the segment we just finished
            if current_segment_length >= min_segment_length:
                # This segment was long enough to count as a real detection
                # Only record it if it's different from the last recorded letter
                # (prevents recording the same letter twice in a row from a long hold)
                if current_letter != last_recorded_letter:
                    stable_letters.append(current_letter)
                    last_recorded_letter = current_letter

            # Start tracking the new letter
            current_letter = letter
            current_segment_length = 1

    # Don't forget the last segment (the loop ends without processing it)
    if current_segment_length >= min_segment_length:
        if current_letter != last_recorded_letter:
            stable_letters.append(current_letter)

    return stable_letters


def get_frame_level_smoothed_letters(predictions, confidence_threshold = None, window_size= None):
    if confidence_threshold is None:
        confidence_threshold = config.CONFIDENCE_THRESHOLD
    if window_size is None:
        window_size = config.WINDOW_SIZE

    if len(predictions) == 0:
        return []

    # Build a dict from frame_index to prediction for fast lookup
    frame_to_prediction = {p["frame_index"]: p for p in predictions}

    # Get all frame indices in order
    all_frame_indices = sorted(frame_to_prediction.keys())

    # Filter by confidence first
    confident_frames = {
        idx: frame_to_prediction[idx]
        for idx in all_frame_indices
        if frame_to_prediction[idx]["confidence"] >= confidence_threshold
    }

    # For frames that passed confidence filtering, apply sliding window
    confident_predictions_list = [confident_frames[idx] for idx in sorted(confident_frames.keys())]
    smoothed = apply_sliding_window(confident_predictions_list, window_size)
    smoothed_by_frame = {p["frame_index"]: p["letter"] for p in smoothed}

    # Build the output list — one entry per original frame
    result = []
    for prediction in predictions:
        frame_index = prediction["frame_index"]
        if frame_index in smoothed_by_frame:
            result.append(smoothed_by_frame[frame_index])
        else:
            result.append(None)  # Low confidence frame

    return result


class LiveTranscriber:
    def __init__(self, window_size = None, min_segment_length = None, confidence_threshold = None):
        self.window_size = window_size or config.WINDOW_SIZE
        self.min_segment_length = min_segment_length or config.MIN_SEGMENT_LENGTH
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD

        # Rolling window of (letter, confidence) for the recent frames
        self.recent_predictions = deque(maxlen=self.window_size)

        # Current stable segment tracking
        self.current_segment_letter = None
        self.current_segment_length = 0
        self.last_committed_letter = None

        # The growing transcription string
        self.transcription = ""

        # What to display on-screen right now (smoothed, not necessarily committed)
        self.display_letter = None
        self.display_confidence = 0.0

    def update(self, letter, confidence):
        if confidence < self.confidence_threshold:
            self.display_letter = None
            self.display_confidence = 0.0
            self._reset_segment()
            return

        self.recent_predictions.append((letter, confidence))

        # Majority vote over the rolling window
        window_letters = [p[0] for p in self.recent_predictions]
        letter_counts = Counter(window_letters)
        smoothed_letter, _ = letter_counts.most_common(1)[0]

        # Average confidence of frames that voted for the winner
        winner_confidences = [p[1] for p in self.recent_predictions if p[0] == smoothed_letter]
        smoothed_confidence = sum(winner_confidences) / len(winner_confidences)

        self.display_letter = smoothed_letter
        self.display_confidence = smoothed_confidence

        # Extend or start a new segment
        if smoothed_letter == self.current_segment_letter:
            self.current_segment_length += 1
        else:
            self.current_segment_letter = smoothed_letter
            self.current_segment_length = 1
            if self.last_committed_letter != smoothed_letter:
                self.last_committed_letter = None

        # Commit when the segment first reaches the minimum length
        if (self.current_segment_length == self.min_segment_length and
                smoothed_letter != self.last_committed_letter):
            self.transcription += smoothed_letter
            self.last_committed_letter = smoothed_letter

    def update_no_hand(self):
        self.display_letter = None
        self.display_confidence = 0.0
        self.recent_predictions.clear()
        self._reset_segment()

    def add_space(self):
        self.transcription += " "

    def backspace(self):
        if self.transcription:
            self.transcription = self.transcription[:-1]

    def clear(self):
        self.transcription = ""
        self.recent_predictions.clear()
        self._reset_segment()
        self.display_letter = None
        self.display_confidence = 0.0

    def _reset_segment(self):
        self.current_segment_letter = None
        self.current_segment_length = 0
        self.last_committed_letter = None
