#!/usr/bin/env python3
"""
check_url.py
------------
Quick command-line interface to the trained model.

Usage:
    python check_url.py "http://example.com/login"
    python check_url.py "http://example.com/login" --offline   # skip live SSL/WHOIS checks
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import PhishingDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Check a URL for phishing indicators")
    parser.add_argument("url", help="The URL to analyze")
    parser.add_argument("--offline", action="store_true",
                         help="Skip live SSL/WHOIS/redirect checks (fully offline)")
    args = parser.parse_args()

    detector = PhishingDetector()
    result = detector.analyze(args.url, check_network=not args.offline)
    result.pop("raw_features", None)

    print("\n===================================")
    print(f" URL      : {result['url']}")
    print(f" Verdict  : {result['verdict'].upper()}")
    print(f" Risk     : {result['risk_score']}%")
    print(f" Model    : {result['model_used']}")
    print("-----------------------------------")
    print(" Indicators:")
    for ind in result["indicators"]:
        print(f"   - {ind}")
    print("===================================\n")


if __name__ == "__main__":
    main()
