"""
live_webcam.py — Real-time ASL fingerspelling transcription via webcam.

Opens a webcam window and transcribes ASL letters as you sign them.
The predicted letter and growing transcription are overlaid on the live feed.

Controls:
  SPACE     — Insert a space into the transcription
  BACKSPACE — Delete the last character
  C         — Clear the entire transcription
  Q or ESC  — Quit

Usage:
    python src/live_webcam.py
    python src/live_webcam.py --camera 1   # Use a different camera
"""

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
import argparse
import time
from collections import Counter
import config
from modules.landmark_extractor import create_video_extractor
from modules.feature_engineer import compute_features
from modules.classifier import ASLClassifier
from modules.temporal_aggregation import LiveTranscriber


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time ASL fingerspelling transcription via webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 for built-in webcam)."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Minimum classifier confidence to display a prediction (default: {config.CONFIDENCE_THRESHOLD})."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=config.WINDOW_SIZE,
        help=f"Sliding window size for majority vote smoothing (default: {config.WINDOW_SIZE})."
    )
    parser.add_argument(
        "--min-segment",
        type=int,
        default=config.MIN_SEGMENT_LENGTH,
        help=f"Minimum frames a letter must hold to be committed (default: {config.MIN_SEGMENT_LENGTH})."
    )
    return parser.parse_args()


def draw_overlay(frame, transcriber, landmarks, fps):
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # Draw semi-transparent dark banner at the top for the transcription text
    banner_height = 80
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Transcription text in the banner
    transcription_display = transcriber.transcription if transcriber.transcription else "Start signing..."
    # Truncate from the left if too long to fit
    max_chars = 35
    if len(transcription_display) > max_chars:
        transcription_display = "..." + transcription_display[-(max_chars - 3):]

    cv2.putText(
        annotated, transcription_display,
        org=(15, 55),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.3,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    # Draw hand landmarks if detected
    if landmarks is not None:
        _draw_hand_skeleton(annotated, landmarks, width, height)

    # Draw the current predicted letter (large box, bottom-left)
    box_size = 120
    box_x, box_y = 15, height - 15 - box_size

    # Background box
    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (box_x, box_y), (box_x + box_size, box_y + box_size),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, annotated, 0.4, 0, annotated)
    cv2.rectangle(annotated, (box_x, box_y), (box_x + box_size, box_y + box_size),
                  (100, 100, 100), 2)

    if transcriber.display_letter is not None:
        # The letter itself
        cv2.putText(
            annotated, transcriber.display_letter,
            org=(box_x + 18, box_y + 90),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3.5,
            color=(0, 255, 100),
            thickness=5,
            lineType=cv2.LINE_AA
        )

        # Confidence bar below the letter box
        bar_width = box_size
        filled_width = int(bar_width * transcriber.display_confidence)
        bar_y = box_y + box_size + 5

        cv2.rectangle(annotated, (box_x, bar_y), (box_x + bar_width, bar_y + 8),
                      (60, 60, 60), -1)
        # Color: green when confident, yellow when borderline
        bar_color = (0, 220, 0) if transcriber.display_confidence >= 0.8 else (0, 200, 200)
        cv2.rectangle(annotated, (box_x, bar_y), (box_x + filled_width, bar_y + 8),
                      bar_color, -1)

        # Segment progress dots (shows how close to committing the letter)
        if transcriber.current_segment_length > 0:
            dot_total = transcriber.min_segment_length
            dot_filled = min(transcriber.current_segment_length, dot_total)
            dot_y = bar_y + 18
            dot_spacing = box_size // (dot_total + 1)
            for i in range(dot_total):
                dot_x = box_x + dot_spacing * (i + 1)
                dot_color = (0, 255, 100) if i < dot_filled else (60, 60, 60)
                cv2.circle(annotated, (dot_x, dot_y), 5, dot_color, -1)

    else:
        # No hand detected — show placeholder
        cv2.putText(
            annotated, "?",
            org=(box_x + 30, box_y + 85),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3.0,
            color=(80, 80, 80),
            thickness=4,
            lineType=cv2.LINE_AA
        )

    # FPS counter (top-right)
    cv2.putText(
        annotated, f"{fps:.0f} fps",
        org=(width - 90, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(150, 150, 150),
        thickness=1
    )

    # Controls help (bottom-right)
    controls = ["SPACE: space", "BKSP: delete", "C: clear", "Q: quit"]
    for i, control_text in enumerate(controls):
        cv2.putText(
            annotated, control_text,
            org=(width - 165, height - 15 - (len(controls) - 1 - i) * 22),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(140, 140, 140),
            thickness=1
        )

    return annotated


def _draw_hand_skeleton(frame, landmarks, frame_width, frame_height):
    # Bones of the hand: pairs of landmark indices
    connections = [
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # palm
        (1, 2), (2, 3), (3, 4),           # thumb
        (5, 6), (6, 7), (7, 8),           # index
        (9, 10), (10, 11), (11, 12),      # middle
        (13, 14), (14, 15), (15, 16),     # ring
        (17, 18), (18, 19), (19, 20),     # pinky
    ]

    # Convert normalized coordinates to pixel positions
    pixel_coords = []
    for lm_x, lm_y, lm_z in landmarks:
        px = int(lm_x * frame_width)
        py = int(lm_y * frame_height)
        pixel_coords.append((px, py))

    # Draw connecting lines
    for start_idx, end_idx in connections:
        cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx],
                 color=(0, 180, 0), thickness=2)

    # Draw landmark circles
    for idx, (px, py) in enumerate(pixel_coords):
        if idx in config.FINGERTIP_INDICES:
            cv2.circle(frame, (px, py), radius=7, color=(0, 255, 200), thickness=-1)
        elif idx == config.WRIST_INDEX:
            cv2.circle(frame, (px, py), radius=7, color=(200, 200, 0), thickness=-1)
        else:
            cv2.circle(frame, (px, py), radius=4, color=(0, 220, 0), thickness=-1)


def main():
    args = parse_arguments()

    print("=" * 60)
    print("ASL Live Webcam Transcription")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Smoothing window: {args.window} frames")
    print(f"Min segment length: {args.min_segment} frames")
    print()
    print("Controls:")
    print("  SPACE     — Insert a space")
    print("  BACKSPACE — Delete last character")
    print("  C         — Clear transcription")
    print("  Q / ESC   — Quit")
    print()

    # Load classifier
    print("Loading classifier...")
    classifier = ASLClassifier()

    # Open webcam
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"ERROR: Could not open camera {args.camera}.")
        sys.exit(1)

    # Set camera to a reasonable resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print("Starting live transcription...\n")

    # Create MediaPipe video extractor (tracking mode)
    extract_landmarks = create_video_extractor(
        detection_confidence=config.MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=config.MEDIAPIPE_TRACKING_CONFIDENCE
    )

    # Initialize live transcriber
    transcriber = LiveTranscriber(
        window_size=args.window,
        min_segment_length=args.min_segment,
        confidence_threshold=args.confidence
    )

    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    print("Webcam window open. Sign ASL letters to transcribe!")

    while True:
        frame_ok, frame_bgr = camera.read()
        if not frame_ok:
            print("ERROR: Lost camera feed.")
            break

        # Flip horizontally so it acts like a mirror (more natural for signing)
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Extract landmarks
        landmarks = extract_landmarks(frame_rgb)

        if landmarks is not None:
            try:
                feature_vector = compute_features(landmarks)
                prediction = classifier.predict(feature_vector)
                transcriber.update(prediction["letter"], prediction["confidence"])
            except Exception as error:
                print(f"Prediction error: {error}")
                transcriber.update_no_hand()
        else:
            transcriber.update_no_hand()

        # Update FPS counter every 15 frames
        fps_frame_count += 1
        if fps_frame_count >= 15:
            elapsed = time.time() - fps_start_time
            current_fps = fps_frame_count / elapsed if elapsed > 0 else 0
            fps_start_time = time.time()
            fps_frame_count = 0

        # Draw overlays
        display_frame = draw_overlay(frame_bgr, transcriber, landmarks, current_fps)

        # Show the window
        cv2.imshow("ASL Live Transcription. Press Q to quit", display_frame)

        # Handle key presses (waitKey(1) gives ~1ms delay for real-time feel)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
            break
        elif key == ord(" "):
            transcriber.add_space()
        elif key == 8 or key == 127:  # BACKSPACE (8 on Windows, 127 on Mac)
            transcriber.backspace()
        elif key == ord("c") or key == ord("C"):
            transcriber.clear()
            print("Transcription cleared.")

    # Final result
    print(f"\nFinal transcription: '{transcriber.transcription}'")

    # Cleanup
    camera.release()
    extract_landmarks.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()