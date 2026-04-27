import os

## Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "asl_alphabet_train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "asl_alphabet_test")
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")


## Supported letters
SUPPORTED_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
] 

## Feature engineering settings
# Keep z-coordinates by default
DROP_Z = False  

# Fingertip landmark indices (in MediaPipe's 21-point hand model)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# MCP (knuckle) landmark indices — base of each finger
MCP_INDICES = [1, 5, 9, 13, 17]

# PIP (middle joint) landmark indices — middle joint of each finger
PIP_INDICES = [2, 6, 10, 14, 18]

# DIP (second joint) landmark indices
DIP_INDICES = [3, 7, 11, 15, 19]

# Wrist landmark index
WRIST_INDEX = 0

# Middle finger MCP — used as the scale reference landmark
MIDDLE_MCP_INDEX = 9


## MediaPipe confidence thresholds
MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

## Temporal aggregation settings
WINDOW_SIZE = 10
MIN_SEGMENT_LENGTH = 5
CONFIDENCE_THRESHOLD = 0.6
