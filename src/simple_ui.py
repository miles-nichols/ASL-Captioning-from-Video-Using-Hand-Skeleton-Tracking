"""
simple_ui.py - Lightweight desktop UI for live ASL transcription.

Launch with:
    python src/simple_ui.py
"""

import argparse
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from live_webcam import draw_overlay
from modules.classifier import ASLClassifier
from modules.feature_engineer import compute_features
from modules.landmark_extractor import create_video_extractor
from modules.temporal_aggregation import LiveTranscriber


DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simple desktop UI for real-time ASL transcription."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    return parser.parse_args()


class ASLTranscriptionUI:
    def __init__(self, root, camera_index):
        self.root = root
        self.camera_index = camera_index
        self.root.title("ASL Sign Language Transcription")
        self.root.geometry("1280x760")
        self.root.minsize(1180, 720)
        self.root.configure(bg="#f4efe6")

        self.classifier = None
        self.extract_landmarks = None
        self.camera = None
        self.transcriber = None
        self.photo_image = None
        self.frame_job = None

        self.current_fps = 0.0
        self.fps_frame_count = 0
        self.fps_start_time = time.time()

        self.status_var = tk.StringVar(value="Ready to start")
        self.transcription_var = tk.StringVar(value="Start signing when the camera is running.")
        self.letter_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="0%")

        self.confidence_setting = tk.DoubleVar(value=config.CONFIDENCE_THRESHOLD)
        self.window_setting = tk.IntVar(value=config.WINDOW_SIZE)
        self.segment_setting = tk.IntVar(value=config.MIN_SEGMENT_LENGTH)

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#f4efe6")
        style.configure("Card.TFrame", background="#fbf8f2")
        style.configure("Header.TLabel", background="#f4efe6", foreground="#1b2a1f",
                        font=("Georgia", 24, "bold"))
        style.configure("Subtle.TLabel", background="#f4efe6", foreground="#5f655f",
                        font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#fbf8f2", foreground="#1b2a1f",
                        font=("Georgia", 13, "bold"))
        style.configure("Body.TLabel", background="#fbf8f2", foreground="#243127",
                        font=("Segoe UI", 11))

        outer = ttk.Frame(self.root, padding=20, style="Panel.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=5)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer, style="Panel.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="ASL Live Transcription", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="A small desktop control room for the project demo.",
            style="Subtle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        video_card = ttk.Frame(outer, padding=14, style="Card.TFrame")
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        video_card.rowconfigure(1, weight=1)
        video_card.columnconfigure(0, weight=1)

        ttk.Label(video_card, text="Camera Feed", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.video_label = tk.Label(
            video_card,
            text="Camera stopped",
            font=("Segoe UI", 18, "bold"),
            bg="#1f231f",
            fg="#f2efe7",
            width=DISPLAY_WIDTH // 12,
            height=DISPLAY_HEIGHT // 24,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        transcript_card = ttk.Frame(video_card, padding=(0, 14, 0, 0), style="Card.TFrame")
        transcript_card.grid(row=2, column=0, sticky="ew")
        transcript_card.columnconfigure(0, weight=1)

        ttk.Label(transcript_card, text="Transcription", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.transcription_label = tk.Label(
            transcript_card,
            textvariable=self.transcription_var,
            anchor="w",
            justify="left",
            wraplength=880,
            padx=12,
            pady=12,
            bg="#eef2e2",
            fg="#182116",
            font=("Segoe UI", 16, "bold"),
        )
        self.transcription_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        sidebar = ttk.Frame(outer, style="Panel.TFrame")
        sidebar.grid(row=1, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        controls_card = ttk.Frame(sidebar, padding=14, style="Card.TFrame")
        controls_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        controls_card.columnconfigure(0, weight=1)
        controls_card.columnconfigure(1, weight=1)
        controls_card.columnconfigure(2, weight=1)

        ttk.Label(controls_card, text="Controls", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Button(controls_card, text="Start Camera", command=self.start_camera).grid(
            row=1, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(controls_card, text="Stop", command=self.stop_camera).grid(
            row=2, column=0, sticky="ew", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(controls_card, text="Add Space", command=self.add_space).grid(
            row=2, column=1, sticky="ew", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(controls_card, text="Backspace", command=self.backspace).grid(
            row=2, column=2, sticky="ew", pady=(10, 0)
        )
        ttk.Button(controls_card, text="Clear Text", command=self.clear_text).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0)
        )

        metrics_card = ttk.Frame(sidebar, padding=14, style="Card.TFrame")
        metrics_card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        metrics_card.columnconfigure(1, weight=1)

        ttk.Label(metrics_card, text="Live Metrics", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        ttk.Label(metrics_card, text="Letter", style="Body.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics_card, textvariable=self.letter_var, style="Body.TLabel").grid(
            row=1, column=1, sticky="e"
        )
        ttk.Label(metrics_card, text="Confidence", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(metrics_card, textvariable=self.confidence_var, style="Body.TLabel").grid(
            row=2, column=1, sticky="e", pady=(8, 0)
        )
        ttk.Label(metrics_card, text="Status", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(metrics_card, textvariable=self.status_var, style="Body.TLabel").grid(
            row=3, column=1, sticky="e", pady=(8, 0)
        )

        settings_card = ttk.Frame(sidebar, padding=14, style="Card.TFrame")
        settings_card.grid(row=2, column=0, sticky="ew")
        settings_card.columnconfigure(0, weight=1)

        ttk.Label(settings_card, text="Recognition Settings", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self._add_scale(
            settings_card,
            "Confidence threshold",
            self.confidence_setting,
            from_=0.1,
            to=0.95,
            row=1,
        )
        self._add_scale(
            settings_card,
            "Smoothing window",
            self.window_setting,
            from_=3,
            to=20,
            row=2,
        )
        self._add_scale(
            settings_card,
            "Min segment length",
            self.segment_setting,
            from_=2,
            to=12,
            row=3,
        )

        note = (
            "Settings apply when the camera starts. Use the buttons instead of keyboard shortcuts "
            "during the demo."
        )
        ttk.Label(settings_card, text=note, style="Body.TLabel", wraplength=280, justify="left").grid(
            row=4, column=0, sticky="w", pady=(12, 0)
        )

    def _add_scale(self, parent, label, variable, from_, to, row):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)

        ttk.Label(frame, text=label, style="Body.TLabel").grid(row=0, column=0, sticky="w")

        value_label = ttk.Label(frame, text="", style="Body.TLabel")
        value_label.grid(row=0, column=1, sticky="e")

        def sync_label(*_args):
            value = variable.get()
            value_label.config(text=f"{value:.2f}" if isinstance(value, float) else str(value))

        variable.trace_add("write", sync_label)
        sync_label()

        resolution = 0.05 if isinstance(variable, tk.DoubleVar) else 1
        scale = tk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            orient="horizontal",
            resolution=resolution,
            showvalue=False,
            bg="#fbf8f2",
            fg="#243127",
            highlightthickness=0,
            troughcolor="#d8ddc8",
            activebackground="#5d8a50",
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew")

    def start_camera(self):
        if self.camera is not None:
            self.status_var.set("Camera already running")
            return

        self.status_var.set("Loading model...")
        self.root.update_idletasks()

        try:
            self.classifier = ASLClassifier()
            self.extract_landmarks = create_video_extractor(
                detection_confidence=config.MEDIAPIPE_DETECTION_CONFIDENCE,
                tracking_confidence=config.MEDIAPIPE_TRACKING_CONFIDENCE,
            )
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}.")

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            self.transcriber = LiveTranscriber(
                window_size=self.window_setting.get(),
                min_segment_length=self.segment_setting.get(),
                confidence_threshold=self.confidence_setting.get(),
            )

            self.current_fps = 0.0
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
            self.status_var.set("Running")
            self._schedule_next_frame()
        except Exception as error:
            self.stop_camera()
            self.status_var.set("Start failed")
            messagebox.showerror("Camera Error", str(error))

    def stop_camera(self):
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if self.extract_landmarks is not None:
            self.extract_landmarks.close()
            self.extract_landmarks = None

        self.classifier = None
        self.transcriber = None
        self.photo_image = None
        self.video_label.configure(image="", text="Camera stopped")
        self.letter_var.set("-")
        self.confidence_var.set("0%")

        if self.status_var.get() == "Running":
            self.status_var.set("Stopped")

    def add_space(self):
        if self.transcriber is not None:
            self.transcriber.add_space()
            self._sync_text_labels()

    def backspace(self):
        if self.transcriber is not None:
            self.transcriber.backspace()
            self._sync_text_labels()

    def clear_text(self):
        if self.transcriber is not None:
            self.transcriber.clear()
            self._sync_text_labels()

    def _schedule_next_frame(self):
        self.frame_job = self.root.after(15, self._process_frame)

    def _process_frame(self):
        if self.camera is None or self.transcriber is None:
            return

        frame_ok, frame_bgr = self.camera.read()
        if not frame_ok:
            self.status_var.set("Camera feed lost")
            self.stop_camera()
            return

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self.extract_landmarks(frame_rgb)

        if landmarks is not None:
            try:
                feature_vector = compute_features(landmarks)
                prediction = self.classifier.predict(feature_vector)
                self.transcriber.update(prediction["letter"], prediction["confidence"])
            except Exception:
                self.transcriber.update_no_hand()
        else:
            self.transcriber.update_no_hand()

        self.fps_frame_count += 1
        if self.fps_frame_count >= 15:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_frame_count / elapsed if elapsed > 0 else 0.0
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

        display_frame = draw_overlay(frame_bgr, self.transcriber, landmarks, self.current_fps)
        self._render_frame(display_frame)
        self._sync_text_labels()
        self._schedule_next_frame()

    def _render_frame(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        scale = min(DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)
        resized = cv2.resize(
            frame_bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo_image, text="")

    def _sync_text_labels(self):
        if self.transcriber is None:
            self.transcription_var.set("Start signing when the camera is running.")
            self.letter_var.set("-")
            self.confidence_var.set("0%")
            return

        text = self.transcriber.transcription
        self.transcription_var.set(text if text else "Start signing when the camera is running.")
        self.letter_var.set(self.transcriber.display_letter or "-")
        self.confidence_var.set(f"{self.transcriber.display_confidence * 100:.0f}%")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    args = parse_arguments()
    root = tk.Tk()
    app = ASLTranscriptionUI(root, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
