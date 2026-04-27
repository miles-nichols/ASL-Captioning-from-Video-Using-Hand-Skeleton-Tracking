import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.abspath('..'))
from typing import Optional
import config


def load_model(model_path, scaler_path, label_encoder_path):
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    if label_encoder_path is None:
        label_encoder_path = config.LABEL_ENCODER_PATH

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load scaler if it exists
    scaler = None
    if scaler_path is None:
        scaler_path = config.SCALER_PATH

    if os.path.exists(scaler_path):
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("No scaler found — assuming model does not require feature scaling.")

    # Load label encoder
    label_encoder = None
    if os.path.exists(label_encoder_path):
        print(f"Loading label encoder from: {label_encoder_path}")
        label_encoder = joblib.load(label_encoder_path)
    else:
        print("No label encoder found — predictions will be numeric class indices.")

    return model, scaler, label_encoder


def predict_letter(feature_vector, model, scaler, label_encoder):

    # Reshape from 1D (87,) to 2D (1, 87) because scikit-learn expects a 2D array
    features_2d = feature_vector.reshape(1, -1)

    # Apply scaling if a scaler was provided
    if scaler is not None:
        features_2d = scaler.transform(features_2d)

    # Get the predicted class index
    predicted_class_index = model.predict(features_2d)[0]

    # Get probability estimates for all classes
    all_probabilities = {}
    confidence = 1.0  # Default confidence if probabilities aren't available

    if hasattr(model, "predict_proba"):
        probability_array = model.predict_proba(features_2d)[0]
        confidence = float(np.max(probability_array))

        # Map each class index to its letter and probability
        if label_encoder is not None:
            for class_index, probability in enumerate(probability_array):
                letter = label_encoder.inverse_transform([class_index])[0]
                all_probabilities[letter] = float(probability)
        else:
            for class_index, probability in enumerate(probability_array):
                all_probabilities[str(class_index)] = float(probability)

    # Convert class index to letter string
    if label_encoder is not None:
        predicted_letter = label_encoder.inverse_transform([predicted_class_index])[0]
    else:
        predicted_letter = str(predicted_class_index)

    return {
        "letter": predicted_letter,
        "confidence": confidence,
        "all_probabilities": all_probabilities
    }


class ASLClassifier:
    def __init__(self, model_path=None, scaler_path=None, label_encoder_path=None):
        self.model, self.scaler, self.label_encoder = load_model(
            model_path, scaler_path, label_encoder_path
        )
        print("ASLClassifier loaded and ready.")

    def predict(self, feature_vector):
        return predict_letter(
            feature_vector,
            model=self.model,
            scaler=self.scaler,
            label_encoder=self.label_encoder
        )
