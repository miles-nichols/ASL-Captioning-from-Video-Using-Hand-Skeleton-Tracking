# CPRE 5750 sign language transcription

## Simple UI

This project now includes a small desktop UI for live ASL transcription.

Install dependencies first. The project expects the pinned versions in
`requirements.txt` because the trained model and legacy MediaPipe Hands API
are not compatible with the newest package releases.

Run:

```bash
Run with a vm:

deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/simple_ui.py
```


or
```bash
python -m pip uninstall -y mediapipe scikit-learn numpy
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
python src/simple_ui.py
```





The original OpenCV-only demo is still available with:

```bash
python src/live_webcam.py
```

