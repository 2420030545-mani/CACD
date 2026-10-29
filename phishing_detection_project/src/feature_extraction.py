"""
feature_extraction.py
----------------------
Extracts the feature set used throughout the phishing-detection pipeline
(training AND real-time inference use this exact same function, so the
feature vectors are always consistent).

Feature groups (matches the framework described in the paper):
  1. URL / lexical features        -> always available, no network needed
  2. Domain / structural features  -> always available, no network needed
  3. Security features (HTTPS/SSL) -> best-effort live check (optional)
  4. Redirection features          -> best-effort live check (optional)

If `check_network=True` and the live checks fail (offline, blocked,
timeout, invalid domain, etc.) we fall back to safe neutral defaults so
the pipeline never crashes -- it just loses a bit of signal.
"""

import math
import re
import socket
import ssl
from collections import Counter
from urllib.parse import urlparse

# Optional dependency, only needed for live network checks.
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Ordered list of every feature the model is trained on.
# NOTE: order matters -- this list defines the column order used
# everywhere else in the project.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "digit_ratio",
    "num_special_chars",
    "num_subdomains",
    "num_query_params",
    "has_ip_address",
    "has_at_symbol",
    "has_https_token_in_domain",
    "uses_url_shortener",
    "has_suspicious_keywords",
    "suspicious_keyword_count",
    "tld_length",
    "has_port",
    "is_https_scheme",
    "ssl_valid",
    "domain_age_days",
    "redirect_count",
    "has_punycode",
    "entropy",
    "has_double_slash_in_path",
    "has_sensitive_file_ext",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "banking", "confirm",
    "signin", "password", "ebayisapi", "webscr", "security", "suspend",
    "recover", "unlock", "authenticate", "wallet", "invoice", "billing",
    "support",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
}

_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

SENSITIVE_EXTENSIONS = {
    ".exe", ".scr", ".apk", ".bat", ".vbs", ".cmd", ".iso", ".ps1", ".msi", ".jar", ".zip", ".tar", ".gz"
}


def _calculate_entropy(text: str) -> float:
    """Calculates Shannon entropy of the given text string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())
    return round(entropy, 4)



def _get_domain_age_days(hostname: str, timeout: float = 3.0):
    """Best-effort WHOIS lookup. Returns None if unavailable."""
    try:
        import whois  # python-whois, optional dependency
        from datetime import datetime

        w = whois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return None
        if isinstance(created, str):
            return None
        age = (datetime.now() - created).days
        return max(age, 0)
    except Exception:
        return None


def _check_ssl_valid(hostname: str, timeout: float = 3.0):
    """Best-effort TLS handshake check. Returns True/False/None."""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                ssock.getpeercert()
        return True
    except Exception:
        return False


def _check_redirects(url: str, timeout: float = 4.0):
    """Best-effort HTTP redirect count. Returns None if unavailable."""
    if not _HAS_REQUESTS:
        return None
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        return len(resp.history)
    except Exception:
        return None


def extract_features(url: str, check_network: bool = True) -> dict:
    """
    Extract the full feature dictionary for a single URL.

    Parameters
    ----------
    url : str
        The URL to analyze. A scheme (http/https) is added if missing.
    check_network : bool
        If True, attempt live SSL / WHOIS / redirect checks (slower,
        needs internet). If False, those features are filled with
        neutral fallback values (fast, fully offline -- used when
        generating bulk training data).

    Returns
    -------
    dict mapping every name in FEATURE_NAMES -> numeric value.
    """
    raw_url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        raw_url = "http://" + raw_url

    parsed = urlparse(raw_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    features = {}

    features["url_length"] = len(raw_url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)

    features["num_dots"] = raw_url.count(".")
    features["num_hyphens"] = raw_url.count("-")
    features["num_underscores"] = raw_url.count("_")
    features["num_slashes"] = raw_url.count("/")
    digits = sum(c.isdigit() for c in raw_url)
    features["num_digits"] = digits
    features["digit_ratio"] = digits / len(raw_url) if raw_url else 0.0
    features["num_special_chars"] = len(re.findall(r"[^a-zA-Z0-9./:_-]", raw_url))

    host_parts = hostname.split(".") if hostname else []
    # e.g. "a.b.example.com" -> subdomains = a, b  (2)
    features["num_subdomains"] = max(len(host_parts) - 2, 0)

    features["num_query_params"] = len([q for q in query.split("&") if q]) if query else 0

    features["has_ip_address"] = 1 if _IP_PATTERN.match(hostname) else 0

    features["has_at_symbol"] = 1 if "@" in raw_url else 0

    # "https" appearing somewhere in the domain/path (not the scheme) is a
    # classic phishing trick, e.g. http://paypal-secure-https.evil.com
    domain_and_path = hostname + path
    features["has_https_token_in_domain"] = 1 if "https" in domain_and_path.lower() else 0

    root_domain = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else hostname
    features["uses_url_shortener"] = 1 if root_domain.lower() in SHORTENER_DOMAINS else 0

    lower_url = raw_url.lower()
    kw_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in lower_url]
    features["has_suspicious_keywords"] = 1 if kw_hits else 0
    features["suspicious_keyword_count"] = len(kw_hits)

    tld = host_parts[-1] if host_parts else ""
    features["tld_length"] = len(tld)

    features["has_port"] = 1 if parsed.port else 0

    features["is_https_scheme"] = 1 if parsed.scheme == "https" else 0

    # --- network-dependent features ---
    if check_network and hostname:
        ssl_valid = _check_ssl_valid(hostname)
        age_days = _get_domain_age_days(hostname)
        redirects = _check_redirects(raw_url)
    else:
        ssl_valid, age_days, redirects = None, None, None

    features["ssl_valid"] = 1 if ssl_valid else 0 if ssl_valid is False else -1
    features["domain_age_days"] = age_days if age_days is not None else -1
    features["redirect_count"] = redirects if redirects is not None else -1

    # 1. Punycode check (IDN homograph attack, e.g. xn--apple...)
    features["has_punycode"] = 1 if "xn--" in (hostname or "").lower() or "xn--" in raw_url.lower() else 0

    # 2. Shannon entropy (measures randomness/algorithmic generation)
    features["entropy"] = _calculate_entropy(raw_url)

    # 3. Double slash in path (open redirect bypass)
    features["has_double_slash_in_path"] = 1 if "//" in path else 0

    # 4. Sensitive file extensions (.exe, .scr, .apk, .bat, etc.)
    lower_path = path.lower()
    features["has_sensitive_file_ext"] = 1 if any(lower_path.endswith(ext) for ext in SENSITIVE_EXTENSIONS) else 0

    # Ensure fixed order / completeness
    return {name: features.get(name, -1) for name in FEATURE_NAMES}


if __name__ == "__main__":
    import sys
    import json

    test_url = sys.argv[1] if len(sys.argv) > 1 else "http://secure-login-paypal.verify-account.tk/update"
    feats = extract_features(test_url, check_network=False)
    print(json.dumps(feats, indent=2))
