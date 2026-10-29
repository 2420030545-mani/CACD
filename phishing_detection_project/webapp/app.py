"""
app.py
------
Flask web application implementing the "real-time inference" endpoint
of the framework: a user pastes a URL, the site extracts features,
runs the trained model, and returns a legitimate / suspicious /
phishing verdict with a risk score and interpretable indicators.

Run with:
    python webapp/app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import sys

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import PhishingDetector  # noqa: E402

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib")):
            raise RuntimeError(
                "No trained model found. Run this first:\n"
                "  python src/generate_dataset.py\n"
                "  python src/train_model.py"
            )
        _detector = PhishingDetector(models_dir=MODELS_DIR)
    return _detector


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400

    # Live SSL/WHOIS/redirect checks require internet access from this
    # server. Can be controlled via request JSON or NETWORK_FEATURES env var.
    req_network = data.get("check_network")
    if req_network is not None:
        check_network = bool(req_network)
    else:
        check_network = os.environ.get("NETWORK_FEATURES", "1") != "0"

    try:
        detector = get_detector()
        result = detector.analyze(url, check_network=check_network)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Could not analyze URL: {e}"}), 400

    result["features"] = result.pop("raw_features", {})
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
