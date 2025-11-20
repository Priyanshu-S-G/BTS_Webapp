# app/utils.py
"""
Small helper utilities:
- validate uploaded filenames
- parse slice index safely
- load per-model metadata.json (optional)
"""

import os
import json

# allowed MRI file extensions
ALLOWED_EXT = {'.nii', '.nii.gz'}


def allowed_filename(filename: str) -> bool:
    """
    Validate file extension for .nii or .nii.gz.
    """
    fname = filename.lower()
    return any(fname.endswith(ext) for ext in ALLOWED_EXT)


def parse_slice_index(raw):
    """
    Convert user input to int or return None if invalid.
    """
    try:
        return int(raw)
    except:
        return None


def load_model_metadata(model_name: str):
    """
    Try loading models/<model_name>/metadata.json.
    If missing, fall back to defaults that match the notebook.
    """
    meta_path = os.path.join("models", model_name, "metadata.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            pass  # fallback below

    # Defaults inferred from training notebook
    return {
        "input_shape": [128, 128, 4],
        "channel_order": ["flair", "t1", "t1ce", "t2"],
        "preprocess": "minmax_per_channel",
        "threshold": 0.5
    }
