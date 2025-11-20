# app/inference.py
"""
Model loading and inference helpers.

Expectations:
- preprocessed_array: (H, W, 4) float32 in [0,1], channels-last (FLAIR,T1,T1CE,T2)
- model files live under models/<model_name>/ (e.g. models/unet/)
- optional per-model custom_objects can be provided at models/<model_name>/custom_objects.py
- metadata can be read via utils.load_model_metadata(model_name) for threshold/input shape info

Notebook reference (uploaded): /mnt/data/BraTS_kaggle.ipynb
"""

import io
import os
import base64
from functools import lru_cache
import importlib.util

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

import utils  # relative import from app package

# If you want a quick reference to the notebook used for training/debug:
NOTEBOOK_PATH = "/mnt/data/BraTS_kaggle.ipynb"

# Top-level models dir (relative to repo root)
MODELS_ROOT = os.path.join("models")


def _load_custom_objects_if_present(model_name: str):
    """
    If models/<model_name>/custom_objects.py exists, import it and return dict of symbols.
    The file should expose a dict named CUSTOM_OBJECTS or individual functions/cls.
    """
    custom_path = os.path.join(MODELS_ROOT, model_name, "custom_objects.py")
    if not os.path.exists(custom_path):
        return {}

    spec = importlib.util.spec_from_file_location(f"{model_name}_custom_objects", custom_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Prefer explicit dict
    if hasattr(mod, "CUSTOM_OBJECTS") and isinstance(mod.CUSTOM_OBJECTS, dict):
        return mod.CUSTOM_OBJECTS
    # Otherwise collect callables/attrs as potential custom objects
    objs = {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}
    return objs


@lru_cache(maxsize=3)
def load_keras_model(model_name: str):
    """
    Load and cache a Keras model for inference.
    Looks for common filenames inside models/<model_name>/ (model.keras, model.h5, <model_name>.h5).
    If custom_objects.py exists under the model folder, it will be used.
    """
    folder = os.path.join(MODELS_ROOT, model_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Model folder not found: {folder}")

    # possible model filenames (in order)
    candidates = ["model.keras", "model.h5", "model.hdf5", f"{model_name}.h5", f"{model_name}.keras"]
    model_path = None
    for cand in candidates:
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(f"No model file found for {model_name} in {folder}. Searched: {candidates}")

    # load custom objects if present
    custom_objs = _load_custom_objects_if_present(model_name)
    if custom_objs:
        return load_model(model_path, compile=False, custom_objects=custom_objs)
    else:
        return load_model(model_path, compile=False)


def predict_mask(model_name: str, preprocessed_array: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Run model inference on a single preprocessed array.
    - model_name: name of the folder under models/
    - preprocessed_array: (H,W,4) float32 in [0,1]
    Returns:
    - binmask_uint8: (H,W) uint8 array with values {0,255}
    """
    if preprocessed_array.ndim != 3 or preprocessed_array.shape[-1] != 4:
        raise ValueError("preprocessed_array must be (H,W,4) channels-last")

    # load metadata default threshold if not provided
    meta = utils.load_model_metadata(model_name)
    used_threshold = threshold if threshold is not None else float(meta.get("threshold", 0.5))

    model = load_keras_model(model_name)
    x = np.expand_dims(preprocessed_array, axis=0)  # (1,H,W,4)

    pred = model.predict(x)
    # normalize pred shape handling:
    # common shapes: (1,H,W,1), (1,H,W), (1,H,W,C)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        mask = pred[0, ..., 0]
    elif pred.ndim == 3:
        # shape (1,H,W) or (1,H,W,C) collapsed; if C>1, take first channel
        mask = pred[0]
        if mask.ndim == 3 and mask.shape[-1] > 1:
            mask = mask[..., 0]
    else:
        # fallback: take first channel of first batch
        mask = pred[0, ..., 0] if pred.ndim >= 3 else pred[0]

    # Threshold to binary
    binmask = (mask > used_threshold).astype("uint8") * 255
    return binmask


def mask_to_png_b64(mask_uint8: np.ndarray) -> str:
    """
    Convert a 2D uint8 mask (0/255) to base64 PNG string.
    """
    if mask_uint8.dtype != np.uint8:
        mask_uint8 = mask_uint8.astype("uint8")
    im = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview_rgb_b64(preprocessed_array: np.ndarray) -> str:
    """
    Create a 3-channel RGB preview PNG from the first 3 channels of the preprocessed array.
    Expects input (H,W,4) float [0,1]. Returns base64 PNG string.
    """
    if preprocessed_array.ndim != 3 or preprocessed_array.shape[-1] < 3:
        raise ValueError("preprocessed_array must have at least 3 channels for preview")

    arr = (np.clip(preprocessed_array[..., :3], 0.0, 1.0) * 255.0).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Example quick test (not executed here):
# from app.preprocess import build_preprocessed_from_files
# arr, used_slice = build_preprocessed_from_files(uploaded_files, slice_index=None)
# mask = predict_mask("unet", arr)
# mask_b64 = mask_to_png_b64(mask)
# preview_b64 = preview_rgb_b64(arr)
# return {"preview_b64": preview_b64, "mask_b64": mask_b64, "used_slice": used_slice}
