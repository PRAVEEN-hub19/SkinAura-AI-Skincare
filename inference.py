"""Shared model loading and prediction utilities for SkinAura interfaces."""

import json
import os

import numpy as np
from PIL import Image

import config

_model = None
_index_to_label = None


def _raise_runtime_import_error(error):
    detail = str(error)
    normalized_detail = detail.casefold()
    if "application control policy" in normalized_detail or "app control policy" in normalized_detail:
        raise RuntimeError(
            "Windows Application Control blocked a native TensorFlow dependency, so SkinAura "
            "cannot run image analysis. Ask your administrator to approve the dependency or "
            "provide an approved Python environment. Do not disable security controls. "
            f"Original error: {detail}"
        ) from error
    if isinstance(error, ModuleNotFoundError) and error.name == "optree":
        raise RuntimeError(
            "Keras requires the optree package. Install it in the active Python environment "
            "with `python -m pip install -r requirements.txt`."
        ) from error
    if "optree installed" in normalized_detail:
        raise RuntimeError(
            "Keras requires the optree package. Install it in the active Python environment "
            "with `python -m pip install -r requirements.txt`."
        ) from error
    raise RuntimeError(
        "TensorFlow's native runtime could not be loaded. Reinstall the project dependencies "
        f"in the active Python environment and try again. Original error: {detail}"
    ) from error


def load_model_and_labels():
    """Load the trained model once, on the first prediction request."""
    global _model, _index_to_label

    if _model is not None:
        return
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_PATH}.\n\n"
            "This file is created after training. Make sure the dataset is "
            "inside dataset\\train\\dry, dataset\\train\\normal, and "
            "dataset\\train\\oily, then run:\npython train_model.py"
        )
    if not os.path.exists(config.LABELS_PATH):
        raise FileNotFoundError(
            f"Label file not found at {config.LABELS_PATH}.\nRun train_model.py first."
        )

    try:
        import optree  # noqa: F401
        from tensorflow.keras.models import load_model
    except (ImportError, OSError) as error:
        _raise_runtime_import_error(error)

    _model = load_model(config.MODEL_PATH)
    with open(config.LABELS_PATH, "r", encoding="utf-8") as labels_file:
        label_to_index = json.load(labels_file)
    _index_to_label = {value: key for key, value in label_to_index.items()}


def _prepare_image(image):
    input_shape = _model.input_shape
    image_size = tuple(input_shape[1:3]) if input_shape else config.IMG_SIZE
    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image).astype("float32")
    if _model.name != "SkinAura_MobileNetV2":
        array /= 255.0
    return np.expand_dims(array, axis=0)


def predict_skin_type(image):
    """Return the predicted label and confidence for a PIL image or path."""
    prediction = predict_skin_type_details(image)[0]
    return prediction["label"], prediction["confidence"]


def predict_skin_type_details(image):
    """Return all class probabilities, ranked for this particular image."""
    load_model_and_labels()
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    predictions = _model.predict(_prepare_image(image), verbose=0)[0]
    ranked_indices = np.argsort(predictions)[::-1]
    return [
        {
            "label": _index_to_label[int(index)],
            "confidence": float(predictions[index]) * 100.0,
        }
        for index in ranked_indices
    ]


def get_model_warning():
    """Explain when predictions come from the bundled pre-fix model."""
    load_model_and_labels()
    if _model.name != "SkinAura_MobileNetV2":
        return (
            "Legacy model: this CNN predates the corrected training workflow. "
            "Treat its prediction as experimental; retrain after adding all "
            "three classes under train/."
        )
    return None
