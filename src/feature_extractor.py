"""Safe, offline lexical feature extraction for a user-entered URL.

This module never opens, requests, or visits the supplied URL. It only parses
the text so it can be passed to the trained machine-learning model.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

FEATURE_COLUMNS = [
    "url_length", "has_ip_address", "dot_count", "https_flag", "url_entropy",
    "token_count", "subdomain_count", "query_param_count", "tld_length",
    "path_length", "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
    "suspicious_file_extension", "domain_name_length", "percentage_numeric_chars",
]

FEATURE_NAMES = {
    "url_length": "URL length", "has_ip_address": "IP address in URL",
    "dot_count": "Number of dots", "https_flag": "HTTPS used",
    "url_entropy": "URL entropy", "token_count": "URL token count",
    "subdomain_count": "Subdomain count", "query_param_count": "Query parameter count",
    "tld_length": "TLD length", "path_length": "Path length",
    "has_hyphen_in_domain": "Hyphen in domain", "number_of_digits": "Number of digits",
    "tld_popularity": "Popular TLD", "suspicious_file_extension": "Suspicious file extension",
    "domain_name_length": "Domain name length", "percentage_numeric_chars": "Numeric-character percentage",
}

# This deliberately small, documented list can be adjusted later if project
# requirements specify a different definition of "popular".
POPULAR_TLDS = {"com", "org", "net", "edu", "gov"}
SUSPICIOUS_EXTENSIONS = {".exe", ".zip", ".bin", ".apk", ".cmd", ".js"}


def _parse_url(raw_url: str):
    """Parse a URL even when a user omits its scheme."""
    candidate = raw_url.strip()
    if not candidate:
        raise ValueError("Please enter a URL.")
    # Parse the raw input for feature compatibility. The training dataset used
    # urlparse on raw strings, including URLs with no scheme; that convention
    # yields a blank netloc and a full-string path for those entries.
    parsed = urlsplit(candidate)
    validation_parse = parsed if parsed.hostname else urlsplit("//" + candidate)
    if not validation_parse.hostname:
        raise ValueError("Enter a valid URL with a domain name or IP address.")
    return candidate, parsed


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict[str, float | int]:
    """Return the 16 model features in the exact training-column order."""
    raw_url, parsed = _parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    # netloc retains a port, matching the simple host-token convention used by
    # the dataset for IP-address URLs.
    netloc_parts = parsed.netloc.lower().rsplit("@", 1)[-1].split(".")
    domain_parts = hostname.split(".")
    is_ip = _is_ip_address(hostname)
    tld_token = netloc_parts[-1] if netloc_parts else ""
    domain_token = netloc_parts[-2] if len(netloc_parts) >= 2 else hostname
    # Dataset behavior treats a URL with no query string as one query segment.
    query_parameter_count = max(1, len(parsed.query.split("&")))
    # This mirrors the source dataset's token rule exactly: split on each URL
    # separator and retain empty segments such as the one in "https://".
    tokens = re.split(r"[./?&=]", raw_url)
    path_lower = parsed.path.lower()

    features = {
        "url_length": len(raw_url),
        "has_ip_address": int(is_ip),
        "dot_count": raw_url.count("."),
        "https_flag": int(raw_url.lower().startswith("https://")),
        "url_entropy": _entropy(raw_url),
        "token_count": len(tokens),
        "subdomain_count": len(domain_parts) - 2,
        "query_param_count": query_parameter_count,
        "tld_length": len(tld_token),
        "path_length": len(parsed.path),
        "has_hyphen_in_domain": int("-" in hostname),
        "number_of_digits": sum(character.isdigit() for character in raw_url),
        "tld_popularity": int(tld_token in POPULAR_TLDS),
        "suspicious_file_extension": int(any(path_lower.endswith(extension) for extension in SUSPICIOUS_EXTENSIONS)),
        "domain_name_length": len(domain_token),
        "percentage_numeric_chars": (sum(character.isdigit() for character in raw_url) / len(raw_url)) * 100,
    }
    return {column: features[column] for column in FEATURE_COLUMNS}


def features_as_dataframe(url: str) -> pd.DataFrame:
    """Create a one-row DataFrame compatible with the saved model pipeline."""
    return pd.DataFrame([extract_features(url)], columns=FEATURE_COLUMNS)
