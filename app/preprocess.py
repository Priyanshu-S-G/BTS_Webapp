# app/preprocess.py
"""
Preprocessing utilities for inference.

Expected modality upload order (BRATS default):
    flair, t1, t1ce, t2

Main functions:
- load_volume(fileobj) -> np.ndarray (H,W,Z)
- extract_slice_from_volumes(volumes_dict, slice_index=None) -> (H,W,4), used_slice
- resize_and_normalize(img_hw4, target=(128,128)) -> (128,128,4) float32 in [0,1]
- build_preprocessed_from_files(file_dict, slice_index=None, target=(128,128)) -> (128,128,4), used_slice
"""
import io
import os
import tempfile
from typing import Dict, Tuple
import numpy as np
import nibabel as nib
import cv2

# Enforce the channel order used during training
CHANNEL_ORDER = ["flair", "t1", "t1ce", "t2"]


def load_volume(fileobj) -> np.ndarray:
    """
    Load a NIfTI volume from a file-like object or path.
    - If fileobj has a .read() (uploaded FileStorage), write to a temporary file first
      because nibabel on some platforms expects a real filename.
    - If fileobj is a path-like (str / Path), load directly.

    Returns numpy array (H, W, Z) or (H, W) for single-slice files.
    """
    # If it's a path or path-like, let nibabel load it directly
    if isinstance(fileobj, (str, os.PathLike)):
        data = nib.load(fileobj).get_fdata()
        return np.asarray(data)

    # If it's a Flask FileStorage or any file-like object, try to use .save if available
    tmp_path = None
    try:
        # Werkzeug FileStorage has .save(filename)
        if hasattr(fileobj, "save"):
            # create a temp file with .nii suffix
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
            tmp_path = tf.name
            tf.close()
            fileobj.save(tmp_path)
            data = nib.load(tmp_path).get_fdata()
            return np.asarray(data)

        # Otherwise, try to read bytes and write to temp file
        if hasattr(fileobj, "read"):
            bytes_data = fileobj.read()
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
            tmp_path = tf.name
            with open(tmp_path, "wb") as f:
                f.write(bytes_data)
            data = nib.load(tmp_path).get_fdata()
            return np.asarray(data)

        # fallback: try nibabel directly (may fail)
        data = nib.load(fileobj).get_fdata()
        return np.asarray(data)

    finally:
        # cleanup temp file if we created one
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_slice_from_volumes(volumes: Dict[str, np.ndarray], slice_index: int = None) -> Tuple[np.ndarray, int]:
    """
    Given a dict of modality volumes {name: ndarray(H,W,Z)}, extract the same slice index
    from each modality and return a stacked (H,W,4) array (channels-last).
    If slice_index is None, uses center slice (Z//2).
    """
    # Validate presence of expected channels
    for ch in CHANNEL_ORDER:
        if ch not in volumes:
            raise ValueError(f"Missing modality volume: {ch}")

    # Use the Z dimension of the first modality (assume consistent)
    any_vol = volumes[CHANNEL_ORDER[0]]
    if any_vol.ndim < 3:
        # If input is already 2D (H,W), treat Z=1
        zdim = 1
    else:
        zdim = any_vol.shape[2]

    if slice_index is None:
        slice_index = zdim // 2

    out_ch = []
    for ch in CHANNEL_ORDER:
        vol = volumes[ch]
        # If volume is 2D, just take it
        if vol.ndim == 2:
            sl = vol
        else:
            if slice_index < 0 or slice_index >= vol.shape[2]:
                raise IndexError(f"slice_index {slice_index} out of range for modality {ch} with z={vol.shape[2]}")
            sl = vol[:, :, slice_index]
        out_ch.append(sl)

    stacked = np.stack(out_ch, axis=-1)  # shape (H, W, 4)
    return stacked, int(slice_index)


def resize_and_normalize(img_hw4: np.ndarray, target: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Resize each channel to target (H,W) and apply per-channel min-max normalization.
    Returns float32 array in [0,1] with shape (target_h, target_w, 4).
    """
    tgt_h, tgt_w = target
    out = np.zeros((tgt_h, tgt_w, 4), dtype=np.float32)

    for c in range(4):
        ch = img_hw4[..., c].astype(np.float32)
        # cv2.resize expects (width, height)
        resized = cv2.resize(ch, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

        mn = resized.min()
        mx = resized.max()
        if mx - mn < 1e-8:
            norm = np.zeros_like(resized, dtype=np.float32)
        else:
            norm = (resized - mn) / (mx - mn + 1e-8)
        out[..., c] = norm

    return out


def build_preprocessed_from_files(file_dict: Dict[str, object], slice_index: int = None,
                                  target: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, int]:
    """
    High-level helper that:
    - accepts file_dict mapping modality names to file-like objects:
        {'flair': FileStorage, 't1': FileStorage, 't1ce': ..., 't2': ...}
    - loads volumes, extracts common slice, resizes & normalizes, returns (128,128,4), used_slice

    Raises informative errors on missing files or shape issues.
    """
    # Load volumes
    volumes = {}
    for ch in CHANNEL_ORDER:
        if ch not in file_dict:
            raise ValueError(f"Missing uploaded file for modality: {ch}")
        volumes[ch] = load_volume(file_dict[ch])

    stacked, used_slice = extract_slice_from_volumes(volumes, slice_index=slice_index)
    preproc = resize_and_normalize(stacked, target=target)
    return preproc, used_slice


# Example usage (for reference):
# files = {'flair': request.files['flair'], 't1': request.files['t1'], ...}
# arr, used_slice = build_preprocessed_from_files(files, slice_index=None, target=(128,128))
# Now arr.shape == (128,128,4), dtype float32, values in [0,1]
# Notebook reference: /mnt/data/BraTS_kaggle.ipynb
