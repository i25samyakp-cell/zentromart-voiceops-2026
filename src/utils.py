"""Text, date, privacy, and file helpers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[\w₹]+", re.UNICODE)
SKU_RE = re.compile(r"\b(?:ELE|FSH|HOM|BEA|BKS|GRO|INR|DIG)-?\d{4}\b", re.I)

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [token for token in TOKEN_RE.findall(normalized) if len(token) > 1 or token.isdigit()]


def digits_only(text: str) -> str:
    digits: list[str] = []
    for char in unicodedata.normalize("NFKC", str(text)):
        if char.isdigit():
            try:
                digits.append(str(unicodedata.digit(char)))
            except (TypeError, ValueError):
                digits.append(char)
    return "".join(digits)


def expand_spoken_digits(text: str) -> str:
    """Turn sequences such as 'eight nine seven eight' into 8978."""
    parts = re.findall(r"\w+|[^\w\s]+|\s+", text, flags=re.UNICODE)
    out: list[str] = []
    digit_run: list[str] = []

    def flush() -> None:
        if digit_run:
            out.append("".join(digit_run))
            digit_run.clear()

    for part in parts:
        key = part.casefold()
        if key in _DIGIT_WORDS:
            digit_run.append(_DIGIT_WORDS[key])
        elif part.isspace() and digit_run:
            continue
        else:
            flush()
            out.append(part)
    flush()
    return "".join(out)


def extract_order_id(text: str) -> str | None:
    expanded = expand_spoken_digits(text)
    for match in re.finditer(r"(?:\d[\s-]*){8}", expanded):
        value = digits_only(match.group())
        if len(value) == 8:
            return value
    return None


def extract_verification_value(text: str) -> str | None:
    expanded = expand_spoken_digits(text)
    candidates = re.findall(r"(?<!\d)(?:\d[\s-]*){4,6}(?!\d)", expanded)
    values = [digits_only(item) for item in candidates]
    values = [item for item in values if len(item) in (4, 6)]
    return values[-1] if values else None


def extract_skus(text: str) -> list[str]:
    expanded = expand_spoken_digits(text).upper()
    expanded = re.sub(r"\b(ELE|FSH|HOM|BEA|BKS|GRO|INR|DIG)\s*[- ]?\s*(\d{4})\b", r"\1-\2", expanded)
    return list(dict.fromkeys(match.upper() for match in SKU_RE.findall(expanded)))


def detect_reply_language(text: str) -> str:
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    hinglish = {
        "hai", "hain", "kya", "mujhe", "mera", "meri", "wala", "wali",
        "kaise", "kab", "tak", "pe", "kar", "nahi", "acha", "haan",
    }
    tokens = set(tokenize(text))
    return "hi-en" if len(tokens & hinglish) >= 2 else "en"


def add_business_days(start: date, count: int, holidays: Iterable[date]) -> date:
    holiday_set = set(holidays)
    current = start
    added = 0
    while added < count:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holiday_set:
            added += 1
    return current


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "item"


def redact_order(order: dict | None) -> dict | None:
    """Remove fields that should not be sent to the language model."""
    if not order:
        return None
    blocked = {"phone", "email", "address", "delivery_pincode"}
    return {key: value for key, value in order.items() if key not in blocked}
