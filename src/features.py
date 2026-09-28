"""URL-only feature extraction.

The PhiUSIIL dataset contains both URL and webpage-derived attributes.
This module implements a reproducible URL-only subset so the exact same
feature-generation logic can be used during training and dashboard inference.

The extractor intentionally does not visit or execute the submitted website.
All features below are derived from the URL string itself.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import pandas as pd


# ---------------------------------------------------------------------------
# URL-only feature set
# ---------------------------------------------------------------------------
# These features can be generated directly from a submitted URL without
# visiting the destination website.
FEATURE_COLUMNS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "CharContinuationRate",
]


# ---------------------------------------------------------------------------
# Regular expression used to detect percent-encoded characters.
# Example: %20, %2F, %3D
# ---------------------------------------------------------------------------
OBFUSCATION_RE = re.compile(r"%(?:[0-9a-fA-F]{2})")


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------
def _normalise_url(url: str) -> str:
    """Ensure the URL has a scheme before parsing.

    If the user enters:
        example.com/login

    it is internally interpreted as:
        http://example.com/login

    This is only for parsing and does not result in a network request.
    """
    value = str(url).strip()

    if not value:
        return ""

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "http://" + value

    return value


# ---------------------------------------------------------------------------
# Safe hostname extraction
# ---------------------------------------------------------------------------
def _safe_domain(parsed) -> str:
    """Safely obtain the hostname from a parsed URL."""
    try:
        return parsed.hostname or ""
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# IP-address detection
# ---------------------------------------------------------------------------
def _is_ip(domain: str) -> int:
    """Return 1 if the hostname is an IPv4/IPv6 address, otherwise 0."""
    if not domain:
        return 0

    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# TLD length
# ---------------------------------------------------------------------------
def _tld_length(domain: str) -> int:
    """Return the character length of the top-level domain."""
    if not domain or _is_ip(domain):
        return 0

    parts = domain.lower().strip(".").split(".")

    return len(parts[-1]) if parts else 0


# ---------------------------------------------------------------------------
# Subdomain count
# ---------------------------------------------------------------------------
def _subdomain_count(domain: str) -> int:
    """Count subdomain levels in the hostname.

    Example:
        login.account.example.com

    is interpreted as having two subdomain levels:
        login
        account
    """
    if not domain or _is_ip(domain):
        return 0

    parts = [
        part
        for part in domain.lower().strip(".").split(".")
        if part
    ]

    return max(len(parts) - 2, 0)


# ---------------------------------------------------------------------------
# Character continuation rate
# ---------------------------------------------------------------------------
def _continuation_rate(value: str) -> float:
    """Calculate the ratio of adjacent alphanumeric characters.

    The calculation is performed directly on the raw URL so that the same
    logic can be reproduced during dashboard inference.
    """
    if len(value) < 2:
        return 0.0

    continuation = sum(
        a.isalnum() and b.isalnum()
        for a, b in zip(value, value[1:])
    )

    return continuation / max(len(value) - 1, 1)


# ---------------------------------------------------------------------------
# Other special-character count
# ---------------------------------------------------------------------------
def _other_special_count(value: str) -> int:
    """Count uncommon characters while excluding normal URL syntax."""
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ":/?&=#._-~%"
    )

    return sum(1 for char in value if char not in allowed)


# ---------------------------------------------------------------------------
# Main URL feature extractor
# ---------------------------------------------------------------------------
def extract_url_features(url: str) -> dict[str, float | int]:
    """Extract the URL-only features used by the ML model.

    Parameters
    ----------
    url:
        URL supplied by the user.

    Returns
    -------
    dict
        Dictionary containing the same URL-derived features used during
        model training.

    Important
    ---------
    This function does NOT make a network request and does not visit the
    submitted website.
    """

    raw = str(url).strip()

    normalised = _normalise_url(raw)
    parsed = urlsplit(normalised)

    domain = _safe_domain(parsed)

    # ---------------------------------------------------------------
    # Character-level measurements
    # ---------------------------------------------------------------
    letters = sum(char.isalpha() for char in raw)
    digits = sum(char.isdigit() for char in raw)

    equals = raw.count("=")
    qmarks = raw.count("?")
    ampersands = raw.count("&")

    other_special = _other_special_count(raw)

    # ---------------------------------------------------------------
    # URL encoding / obfuscation
    # ---------------------------------------------------------------
    obfuscated_matches = OBFUSCATION_RE.findall(raw)

    # Each %XX sequence represents three characters in the raw URL.
    obfuscated_chars = len(obfuscated_matches) * 3

    # Prevent division by zero for empty input.
    length = max(len(raw), 1)

    special_total = (
        equals
        + qmarks
        + ampersands
        + other_special
    )

    # ---------------------------------------------------------------
    # Feature dictionary
    # ---------------------------------------------------------------
    return {
        "URLLength": len(raw),

        "DomainLength": len(domain),

        "IsDomainIP": _is_ip(domain),

        "TLDLength": _tld_length(domain),

        "NoOfSubDomain": _subdomain_count(domain),

        "HasObfuscation": int(
            bool(obfuscated_matches)
            or "@" in raw
            or "\\x" in raw.lower()
        ),

        "NoOfObfuscatedChar": obfuscated_chars,

        "NoOfLettersInURL": letters,

        "LetterRatioInURL": letters / length,

        "NoOfDegitsInURL": digits,

        "DegitRatioInURL": digits / length,

        "NoOfEqualsInURL": equals,

        "NoOfQMarkInURL": qmarks,

        "NoOfAmpersandInURL": ampersands,

        "NoOfOtherSpecialCharsInURL": other_special,

        "SpacialCharRatioInURL": special_total / length,

        "IsHTTPS": int(
            parsed.scheme.lower() == "https"
        ),

        "CharContinuationRate": _continuation_rate(raw),
    }


# ---------------------------------------------------------------------------
# Transform multiple URLs into a model-ready DataFrame
# ---------------------------------------------------------------------------
def transform_urls(urls: pd.Series) -> pd.DataFrame:
    """Convert a pandas Series of URLs into model-ready features.

    The output columns are forced into FEATURE_COLUMNS order so that the
    feature order is identical during training and inference.
    """
    rows = [
        extract_url_features(url)
        for url in urls.astype(str)
    ]

    return (
        pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        .fillna(0)
    )


# ---------------------------------------------------------------------------
# Human-readable feature explanations
# ---------------------------------------------------------------------------
def explain_feature(feature_name: str) -> str:
    """Return a human-readable explanation for a model feature."""

    descriptions = {
        "URLLength":
            "Total length of the submitted URL",

        "DomainLength":
            "Length of the hostname",

        "IsDomainIP":
            "Whether the hostname is an IP address instead of a domain name",

        "TLDLength":
            "Length of the top-level domain",

        "NoOfSubDomain":
            "Number of subdomain levels",

        "HasObfuscation":
            "Detected URL encoding or obfuscation patterns",

        "NoOfObfuscatedChar":
            "Approximate number of characters involved in URL encoding",

        "NoOfLettersInURL":
            "Number of alphabetic characters in the URL",

        "LetterRatioInURL":
            "Proportion of alphabetic characters in the URL",

        "NoOfDegitsInURL":
            "Number of numeric characters in the URL",

        "DegitRatioInURL":
            "Proportion of numeric characters in the URL",

        "NoOfEqualsInURL":
            "Number of '=' query separators",

        "NoOfQMarkInURL":
            "Number of '?' query markers",

        "NoOfAmpersandInURL":
            "Number of '&' query separators",

        "NoOfOtherSpecialCharsInURL":
            "Count of uncommon special characters",

        "SpacialCharRatioInURL":
            "Proportion of special characters in the URL",

        "IsHTTPS":
            "Whether HTTPS is used",

        "CharContinuationRate":
            "Ratio of adjacent alphanumeric characters",
    }

    return descriptions.get(
        feature_name,
        feature_name
    )