# app/app.py
"""
Flask server for quick BRATS slice-level inference demo.

Routes:
- GET  /           -> serves index.html
- POST /predict    -> accepts 4 files (flair,t1,t1ce,t2), model name, optional slice_index
                     returns JSON with preview_b64, mask_b64, and debug info

Run from repo root:
    python app/app.py
"""
import os
import traceback
from flask import Flask, request, jsonify, render_template

# Make sure this file runs when executed from repo root:
# app/ is expected to contain preprocess.py, inference.py, utils.py and templates/static subfolders.
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# local imports (same directory)
from preprocess import build_preprocessed_from_files
from inference import predict_mask, mask_to_png_b64, preview_rgb_b64
import utils

# Allowed model names (folders under /models)
try:
    AVAILABLE_MODELS = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
except Exception:
    # fallback if models folder missing or script run from unexpected cwd
    AVAILABLE_MODELS = ["unet", "segnet", "minisegnet"]

# For a robust check fallback list (will still allow common names)
FALLBACK_MODELS = ["unet", "segnet", "minisegnet"]


@app.route("/")
def index():
    # Render the index page. The frontend will handle upload.
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects multipart/form-data with fields:
      - flair (file)
      - t1 (file)
      - t1ce (file)
      - t2 (file)
      - model (string) optional, defaults to 'unet'
      - slice_index (int) optional
    Returns JSON: { preview_b64, mask_b64, debug }
    """
    try:
        # validate presence of modality files
        expected = ["flair", "t1", "t1ce", "t2"]
        files = {}
        for k in expected:
            f = request.files.get(k)
            if f is None:
                return jsonify({"error": f"Missing required file field: {k}"}), 400
            # basic filename check
            if not utils.allowed_filename(f.filename):
                return jsonify({"error": f"Invalid file extension for {k}: {f.filename}"}), 400
            files[k] = f

        # model selection
        model_name = request.form.get("model", "unet").strip()
        if not model_name:
            model_name = "unet"
        # allow fallback even if AVAILABLE_MODELS detection failed earlier
        if model_name not in AVAILABLE_MODELS and model_name not in FALLBACK_MODELS:
            # don't block — let load_keras_model raise a clear error instead
            pass

        # slice index (optional)
        slice_index = utils.parse_slice_index(request.form.get("slice_index"))

        # build preprocessed array (resized, normalized)
        preproc_arr, used_slice = build_preprocessed_from_files(files, slice_index=slice_index,
                                                                target=tuple(utils.load_model_metadata(model_name)["input_shape"][:2]))

        # run model inference
        bin_mask = predict_mask(model_name, preproc_arr, threshold=utils.load_model_metadata(model_name).get("threshold", 0.5))

        # convert to base64 PNGs (in-memory)
        mask_b64 = mask_to_png_b64(bin_mask)
        preview_b64 = preview_rgb_b64(preproc_arr)

        debug = {
            "model": model_name,
            "input_shape": str(preproc_arr.shape),
            "used_slice": used_slice,
            "preprocess": utils.load_model_metadata(model_name).get("preprocess", "minmax_per_channel"),
            "threshold": utils.load_model_metadata(model_name).get("threshold", 0.5)
        }

        return jsonify({"preview_b64": preview_b64, "mask_b64": mask_b64, "debug": debug})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "internal_server_error", "msg": str(e), "trace": tb}), 500


if __name__ == "__main__":
    # when running app/app.py directly the cwd should be repo root to find models/
    # bind to localhost only for demo
    app.run(host="127.0.0.1", port=5000, debug=True)
