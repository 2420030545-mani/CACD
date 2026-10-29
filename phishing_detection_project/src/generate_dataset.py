"""
generate_dataset.py
--------------------
Builds a labeled training dataset entirely offline.

Two data sources are combined:
  1. A large pool of realistic LEGITIMATE-style URLs (clean domains,
     short paths, HTTPS, no suspicious tokens).
  2. A large pool of realistic PHISHING-style URLs (IP hosts, lookalike
     subdomains, suspicious keywords, hyphenated brand-spoofing domains,
     URL shorteners, long/obfuscated paths).

Every URL is passed through the SAME `extract_features()` function used
at real-time inference, so the trained model sees a feature space
identical to what the web app will hand it later. Network-dependent
features (SSL/WHOIS/redirects) can't be checked for thousands of fake
URLs offline, so for this synthetic set we SIMULATE plausible values for
those columns (legit domains skew toward valid SSL + old domain age;
phishing domains skew toward invalid SSL + young domain age) -- this
keeps the demo self-contained without any internet access.

>>> To train on the REAL dataset used in the paper (UCI PhiUSIIL,
>>> 235,795 URLs / 54 features), download it from:
>>>   https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
>>> and pass its CSV path with --real-csv to train_model.py (see README).
"""

import argparse
import csv
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

random.seed(42)

LEGIT_BRANDS = [
    "google.com", "wikipedia.org", "github.com", "amazon.com", "microsoft.com",
    "apple.com", "nytimes.com", "bbc.com", "linkedin.com", "spotify.com",
    "netflix.com", "dropbox.com", "reddit.com", "stackoverflow.com", "adobe.com",
    "salesforce.com", "shopify.com", "cloudflare.com", "mozilla.org", "python.org",
    "harvard.edu", "stanford.edu", "who.int", "un.org", "gov.uk",
]

LEGIT_PATHS = [
    "", "/", "/about", "/products", "/help", "/contact", "/en/home",
    "/docs/getting-started", "/blog/2025/announcement", "/support/faq",
    "/account/settings", "/search?q=weather", "/news/technology",
]

PHISHING_BRAND_TARGETS = [
    "paypal", "amazon", "apple", "netflix", "microsoft", "bankofamerica",
    "chase", "wellsfargo", "facebook", "instagram", "google", "outlook",
    "irs", "dhl", "fedex", "coinbase", "binance",
]

SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work", "info"]

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "password", "security", "suspend", "recover", "unlock", "wallet",
    "billing", "support",
]


def _random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _random_token(n=6):
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(letters) for _ in range(n))


def generate_legit_urls(n):
    urls = []
    for _ in range(n):
        brand = random.choice(LEGIT_BRANDS)
        path = random.choice(LEGIT_PATHS)
        scheme = "https" if random.random() < 0.92 else "http"
        subdomain = random.choice(["", "www.", "www.", "docs.", "support.", "api.", "blog."])
        query = ""
        if random.random() < 0.25:
            query = f"?q={_random_token(5)}&page={random.randint(1, 5)}"
        url = f"{scheme}://{subdomain}{brand}{path}{query}"
        urls.append(url)
    return urls


def generate_phishing_urls(n):
    urls = []
    for _ in range(n):
        style = random.choice([
            "ip", "lookalike", "shortener", "subdomain_spoof", "suspicious_tld", "brand_spoof",
            "homograph_punycode", "double_slash", "malicious_payload", "high_entropy",
        ])
        target = random.choice(PHISHING_BRAND_TARGETS)
        word = random.choice(SUSPICIOUS_WORDS)
        scheme = "https" if random.random() < 0.72 else "http"

        if style == "ip":
            # Raw IP URLs are mostly http, sometimes with custom port
            port_str = f":{random.choice([8080, 8443, 8000, 8888])}" if random.random() < 0.3 else ""
            url = f"http://{_random_ip()}{port_str}/{target}/{word}.php?id={_random_token(8)}"
        elif style == "lookalike":
            tld = random.choice(SUSPICIOUS_TLDS)
            url = f"{scheme}://{target}-{word}-{_random_token(4)}.{tld}/{word}"
        elif style == "shortener":
            shortener = random.choice(["bit.ly", "tinyurl.com", "cutt.ly", "is.gd"])
            url = f"{scheme}://{shortener}/{_random_token(7)}"
        elif style == "subdomain_spoof":
            tld = random.choice(SUSPICIOUS_TLDS + ["com", "net", "org"])
            url = (
                f"{scheme}://{target}.{word}.{_random_token(5)}-secure.{tld}/"
                f"{word}/{_random_token(6)}"
            )
        elif style == "brand_spoof":
            tld = random.choice(SUSPICIOUS_TLDS + ["com", "online", "site"])
            url = f"{scheme}://{target}-{_random_token(3)}-verification.{tld}/{word}-account"
        elif style == "homograph_punycode":
            tld = random.choice(["com", "org", "net"] + SUSPICIOUS_TLDS)
            url = f"{scheme}://xn--{target}{_random_token(3)}.{tld}/{word}-verify"
        elif style == "double_slash":
            tld = random.choice(SUSPICIOUS_TLDS + ["com", "xyz"])
            url = f"{scheme}://{target}-service.{tld}//bypass/{word}?dest=http://evil-{_random_token(4)}.com"
        elif style == "malicious_payload":
            tld = random.choice(SUSPICIOUS_TLDS + ["top", "click"])
            ext = random.choice(["exe", "scr", "apk", "bat", "zip"])
            url = f"{scheme}://{target}-security-client.{tld}/install/{word}_patch_{_random_token(4)}.{ext}"
        elif style == "high_entropy":
            tld = random.choice(SUSPICIOUS_TLDS)
            url = f"{scheme}://{_random_token(16)}.{target}.{tld}/session?auth_token={_random_token(28)}"
        else:  # suspicious_tld
            tld = random.choice(SUSPICIOUS_TLDS)
            url = f"{scheme}://{target}{word}.{tld}/{_random_token(5)}?{word}=1"

        urls.append(url)
    return urls


def build_dataset(n_per_class=1500):
    legit_urls = generate_legit_urls(n_per_class)
    phishing_urls = generate_phishing_urls(n_per_class)

    rows = []
    for url in legit_urls:
        feats = extract_features(url, check_network=False)
        # Simulate network-derived signals typical of legitimate sites
        # Occasionally WHOIS/network is unavailable (-1)
        if random.random() < 0.2:
            feats["ssl_valid"] = -1
            feats["domain_age_days"] = -1
            feats["redirect_count"] = -1
        else:
            feats["ssl_valid"] = 1 if (feats["is_https_scheme"] == 1 and random.random() < 0.98) else 0
            feats["domain_age_days"] = random.randint(365 * 2, 365 * 20) if random.random() < 0.8 else -1
            feats["redirect_count"] = random.choice([0, 0, 0, 1])
        feats["label"] = 0
        feats["url"] = url
        rows.append(feats)

    for url in phishing_urls:
        feats = extract_features(url, check_network=False)
        # Modern phishing often has valid free SSL (Let's Encrypt / Cloudflare)
        if random.random() < 0.2:
            feats["ssl_valid"] = -1
            feats["domain_age_days"] = -1
            feats["redirect_count"] = -1
        else:
            feats["ssl_valid"] = 1 if (feats["is_https_scheme"] == 1 and random.random() < 0.55) else 0
            feats["domain_age_days"] = random.randint(0, 180) if random.random() < 0.8 else -1
            feats["redirect_count"] = random.choice([0, 1, 1, 2, 3])
        feats["label"] = 1
        feats["url"] = url
        rows.append(feats)

    random.shuffle(rows)
    return rows


def write_csv(rows, out_path):
    fieldnames = FEATURE_NAMES + ["label", "url"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic phishing dataset")
    parser.add_argument("--n-per-class", type=int, default=1200,
                         help="Number of samples to generate per class (default 1200)")
    parser.add_argument("--out", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "data", "phishing_dataset.csv"),
                         help="Output CSV path")
    args = parser.parse_args()

    rows = build_dataset(args.n_per_class)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows ({args.n_per_class} legit / {args.n_per_class} phishing) -> {out_path}")


if __name__ == "__main__":
    main()
