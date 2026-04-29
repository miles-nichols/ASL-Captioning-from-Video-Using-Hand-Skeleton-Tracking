import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))
import cv2
import config
from live_webcam import draw_overlay
from modules.classifier import ASLClassifier
from modules.feature_engineer import compute_features
from modules.landmark_extractor import create_video_extractor
from modules.temporal_aggregation import LiveTranscriber

DEFAULT_INPUT = os.path.join(config.VIDEO_INPUT_DIR, "ASL_alphabet.mp4")
OUTPUT_DIR = config.VIDEO_OUTPUT_DIR


def process_video(input_path: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_transcribed.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    classifier = ASLClassifier()
    extract_landmarks = create_video_extractor(
        detection_confidence=config.MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=config.MEDIAPIPE_TRACKING_CONFIDENCE,
    )
    transcriber = LiveTranscriber()

    frame_index = 0
    print(f"Processing {total_frames} frames at {fps:.1f} fps...")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(frame_rgb)

        if landmarks is not None:
            try:
                features = compute_features(landmarks)
                prediction = classifier.predict(features)
                transcriber.update(prediction["letter"], prediction["confidence"])
            except Exception:
                transcriber.update_no_hand()
        else:
            transcriber.update_no_hand()

        annotated = draw_overlay(frame_bgr, transcriber, landmarks, fps)
        writer.write(annotated)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  {frame_index}/{total_frames}  →  '{transcriber.transcription}'")

    cap.release()
    extract_landmarks.close()
    writer.release()

    print(f"\nOutput saved to: {output_path}")
    return transcriber.transcription


def main():
    parser = argparse.ArgumentParser(description="Transcribe ASL fingerspelling in an .mp4 file.")
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to the input .mp4 file.",
    )
    args = parser.parse_args()

    transcription = process_video(args.input)
    print(f"\nFinal transcription: '{transcription}'")


if __name__ == "__main__":
    main()
