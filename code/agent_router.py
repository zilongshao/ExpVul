from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

from tools import llm_tool


TOP_LEVEL_CWES: Sequence[str] = (
    "CWE-20",
    "CWE-22",
    "CWE-78",
    "CWE-89",
    "CWE-119",
    "CWE-125",
    "CWE-190",
    "CWE-200",
    "CWE-269",
    "CWE-284",
    "CWE-287",
    "CWE-362",
    "CWE-401",
    "CWE-416",
    "CWE-476",
    "CWE-703",
    "CWE-787",
)

SUPPORTED_CWES = set(TOP_LEVEL_CWES)

_FINE_TO_TOP = {
    "CWE-120": "CWE-787",
    "CWE-121": "CWE-787",
    "CWE-122": "CWE-787",
    "CWE-124": "CWE-787",
    "CWE-126": "CWE-125",
    "CWE-127": "CWE-125",
    "CWE-129": "CWE-125",
    "CWE-131": "CWE-190",
    "CWE-189": "CWE-190",
    "CWE-191": "CWE-190",
    "CWE-252": "CWE-703",
    "CWE-754": "CWE-703",
    "CWE-755": "CWE-703",
    "CWE-369": "CWE-190",
    "CWE-674": "CWE-703",
    "CWE-835": "CWE-703",
    "CWE-909": "CWE-703",
    "CWE-269": "CWE-269",
    "CWE-823": "CWE-476",
    "CWE-824": "CWE-476",
    "CWE-457": "CWE-476",
    "CWE-434": "CWE-20",
    "CWE-444": "CWE-20",
    "CWE-94": "CWE-20",
    "CWE-134": "CWE-20",
    "CWE-209": "CWE-200",
    "CWE-532": "CWE-200",
    "CWE-306": "CWE-287",
    "CWE-384": "CWE-287",
    "CWE-613": "CWE-287",
    "CWE-400": "CWE-401",
    "CWE-770": "CWE-401",
    "CWE-399": "CWE-401",
    "CWE-834": "CWE-703",
}

_TOKEN_RULES: Sequence[tuple[str, str]] = (
    ("CWE-787", r"\b(?:strcpy|strcat|sprintf|memcpy|memmove|memset|gets)\s*\("),
    ("CWE-125", r"\[[A-Za-z_]\w*\]"),
    ("CWE-476", r"(?:->|\b(?:malloc|calloc|realloc)\s*\()"),
    ("CWE-190", r"(?:\b(?:malloc|calloc|realloc|new)\s*\([^)]*(?:\*|\+|<<)[^)]*\)|size_max|uint_max|overflow)"),
    ("CWE-401", r"\b(?:malloc|calloc|realloc|new)\s*\("),
    ("CWE-416", r"\b(?:free|delete)\s*\("),
    ("CWE-20", r"\b(?:parse|validate|sanitize|atoi|strtol)\w*\s*\("),
    ("CWE-22", r"(?:\.\./|path|filename|open\s*\()"),
    ("CWE-78", r"(?:system|popen|exec\w*)\s*\("),
    ("CWE-89", r"(?:sql|query|execute)\w*\s*\("),
    ("CWE-703", r"(?:error|status|cleanup|resource|errno)\w*"),
    ("CWE-284", r"(?:permission|privilege|access)\w*"),
    ("CWE-287", r"(?:auth|login|credential|session)\w*"),
    ("CWE-362", r"(?:mutex|lock|atomic|thread|race)\w*"),
    ("CWE-200", r"(?:log|print|debug|error)\w*\s*\("),
)


def _normalise(value: str) -> str:
    value = value.strip().upper()
    if not value:
        return ""
    if not value.startswith("CWE-"):
        value = f"CWE-{value}"
    mapped = _FINE_TO_TOP.get(value, value)
    return mapped if mapped in SUPPORTED_CWES else ""


def _heuristic_route(code: str) -> List[str]:
    scores: Dict[str, int] = {cwe: 0 for cwe in TOP_LEVEL_CWES}
    for cwe, pattern in _TOKEN_RULES:
        if re.search(pattern, code, flags=re.IGNORECASE):
            scores[cwe] += 1
    ranked = sorted(scores.items(), key=lambda item: (-item[1], TOP_LEVEL_CWES.index(item[0])))
    selected = [cwe for cwe, score in ranked if score > 0]
    return selected[:4]


def _extract_candidates(text: str) -> List[str]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            value = json.loads(text[start:end + 1])
        else:
            value = re.findall(r"CWE-\d+", text, flags=re.IGNORECASE)
    except (ValueError, TypeError, json.JSONDecodeError):
        value = re.findall(r"CWE-\d+", text, flags=re.IGNORECASE)
    if not isinstance(value, list):
        value = [value]
    result: List[str] = []
    for item in value:
        normalised = _normalise(str(item))
        if normalised and normalised not in result:
            result.append(normalised)
    return result[:4]


def _router_prompt(code: str) -> str:
    labels = ", ".join(TOP_LEVEL_CWES)
    return (
        "Route the supplied C or C++ function to zero or more relevant security "
        "weakness buckets. Use only the listed buckets and return a JSON array. "
        "Use evidence from syntax and data flow, not file names or external labels.\n\n"
        f"Buckets: {labels}\n\nFunction:\n{code}\n\nOutput:"
    )


def detect_possible_cwes(code: str, mode: str = "full", model: str | None = None) -> List[str]:
    if not code or not code.strip():
        return []
    heuristic = _heuristic_route(code)
    if mode == "heuristic-only":
        return heuristic
    if mode == "no-router":
        return []
    if mode in {"full", "llm-only"} and llm_tool.configured():
        try:
            routed = _extract_candidates(
                llm_tool.chat_completion(_router_prompt(code), model=model, temperature=0.0)
            )
            if routed:
                return routed
        except Exception:
            if mode == "llm-only":
                return []
    return heuristic


def route(code: str, mode: str = "full", model: str | None = None) -> List[str]:
    return detect_possible_cwes(code, mode=mode, model=model)


def get_all_cwe_ids() -> List[str]:
    return list(TOP_LEVEL_CWES)
