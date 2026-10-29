"""
predict.py
----------
Stage E of the pipeline: "Real-Time Classification and Risk Assessment".

Loads the trained model + scaler, extracts features for a submitted
URL, produces a legitimate / suspicious / phishing verdict, a 0-100
risk score, and a short list of the indicators that drove the decision
(interpretable output, as described in the paper).
"""

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Human-readable explanations for each raw feature, used to build the
# "why was this flagged" indicator list.
FEATURE_EXPLANATIONS = {
    "has_ip_address": "URL uses a raw IP address instead of a domain name",
    "has_at_symbol": "URL contains an '@' symbol (classic redirection trick)",
    "has_https_token_in_domain": "The word 'https' appears inside the domain/path (spoofing trick)",
    "uses_url_shortener": "URL was created with a link-shortening service",
    "has_suspicious_keywords": "URL contains suspicious keywords (e.g. 'verify', 'login', 'secure')",
    "is_https_scheme": "Connection is NOT encrypted with HTTPS",
    "ssl_valid": "SSL/TLS certificate could not be validated",
    "domain_age_days": "Domain appears to be very newly registered",
    "num_subdomains": "URL uses an unusually high number of subdomains",
    "num_hyphens": "Domain contains an unusually high number of hyphens",
    "redirect_count": "URL performs multiple redirects before landing",
    "url_length": "URL is unusually long",
    "has_punycode": "Domain uses Punycode ('xn--'), indicative of an IDN homograph spoofing attack",
    "entropy": "URL exhibits abnormally high character entropy, indicating obfuscation or DGA generation",
    "has_double_slash_in_path": "URL contains double slashes ('//') in path for redirection bypass",
    "has_sensitive_file_ext": "URL points to a dangerous executable or script file extension",
}


class PhishingDetector:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        with open(os.path.join(models_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]

    def _top_importances(self, n=8):
        return [name for name, _ in self.metadata["feature_importance_ranking"][:n]]

    def analyze(self, url: str, check_network: bool = True) -> dict:
        raw_features = extract_features(url, check_network=check_network)
        # Order the vector exactly as the model was trained on
        vector = np.array([[raw_features[name] for name in self.feature_names]])
        vector_scaled = self.scaler.transform(vector)

        pred = int(self.model.predict(vector_scaled)[0])
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(vector_scaled)[0][1])
        else:
            proba = float(pred)

        risk_score = round(proba * 100, 1)

        if risk_score >= 70:
            verdict = "phishing"
        elif risk_score >= 35:
            verdict = "suspicious"
        else:
            verdict = "legitimate"

        indicators = self._explain(raw_features)

        return {
            "url": url,
            "verdict": verdict,
            "risk_score": risk_score,
            "model_used": self.metadata["best_model"],
            "indicators": indicators,
            "raw_features": raw_features,
        }

    def _explain(self, raw_features: dict):
        """Rule-based explainability layer combined with model importance
        ranking: flags any risky condition present in this specific URL,
        ordered by how important that feature was during training."""
        flagged = []

        def add(key, cond, msg):
            if cond:
                flagged.append(msg)

        add("has_ip_address", raw_features["has_ip_address"] == 1, FEATURE_EXPLANATIONS["has_ip_address"])
        add("has_at_symbol", raw_features["has_at_symbol"] == 1, FEATURE_EXPLANATIONS["has_at_symbol"])
        add("has_https_token_in_domain", raw_features["has_https_token_in_domain"] == 1,
            FEATURE_EXPLANATIONS["has_https_token_in_domain"])
        add("uses_url_shortener", raw_features["uses_url_shortener"] == 1, FEATURE_EXPLANATIONS["uses_url_shortener"])
        add("has_suspicious_keywords", raw_features["has_suspicious_keywords"] == 1,
            FEATURE_EXPLANATIONS["has_suspicious_keywords"])
        add("is_https_scheme", raw_features["is_https_scheme"] == 0, FEATURE_EXPLANATIONS["is_https_scheme"])
        add("ssl_valid", raw_features["ssl_valid"] == 0, FEATURE_EXPLANATIONS["ssl_valid"])
        add("domain_age_days", 0 <= raw_features["domain_age_days"] < 180,
            FEATURE_EXPLANATIONS["domain_age_days"])
        add("num_subdomains", raw_features["num_subdomains"] >= 3, FEATURE_EXPLANATIONS["num_subdomains"])
        add("num_hyphens", raw_features["num_hyphens"] >= 3, FEATURE_EXPLANATIONS["num_hyphens"])
        add("redirect_count", raw_features.get("redirect_count", -1) >= 2, FEATURE_EXPLANATIONS["redirect_count"])
        add("url_length", raw_features.get("url_length", 0) >= 75, FEATURE_EXPLANATIONS["url_length"])
        add("has_punycode", raw_features.get("has_punycode", 0) == 1, FEATURE_EXPLANATIONS["has_punycode"])
        add("entropy", raw_features.get("entropy", 0.0) >= 4.4, FEATURE_EXPLANATIONS["entropy"])
        add("has_double_slash_in_path", raw_features.get("has_double_slash_in_path", 0) == 1,
            FEATURE_EXPLANATIONS["has_double_slash_in_path"])
        add("has_sensitive_file_ext", raw_features.get("has_sensitive_file_ext", 0) == 1,
            FEATURE_EXPLANATIONS["has_sensitive_file_ext"])

        if not flagged:
            flagged.append("No common phishing indicators were detected in this URL.")
        return flagged


if __name__ == "__main__":
    detector = PhishingDetector()
    url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.5/paypal/verify-account/login.php"
    result = detector.analyze(url, check_network=False)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_features"}, indent=2))
