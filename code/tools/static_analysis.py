from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    import tree_sitter as _tree_sitter
    import tree_sitter_c as _tree_sitter_c
except Exception:
    _tree_sitter = None
    _tree_sitter_c = None


DEFENSE_PATTERNS = (
    "bounds",
    "null",
    "return_value",
    "input_validation",
    "safe_api",
    "error_handling",
    "overflow",
)

_UNSAFE_CALLS = {
    "gets": "unbounded input",
    "strcpy": "unbounded copy",
    "strcat": "unbounded concatenation",
    "sprintf": "unbounded formatting",
    "vsprintf": "unbounded formatting",
    "memcpy": "size requires validation",
    "memmove": "size requires validation",
    "memset": "size requires validation",
    "strncpy": "termination depends on the bound",
    "strncat": "destination bound requires validation",
}

_SAFE_CALLS = {
    "snprintf",
    "vsnprintf",
    "strlcpy",
    "strlcat",
    "memcpy_s",
    "memmove_s",
    "memset_s",
    "strcpy_s",
    "strcat_s",
    "strtol",
    "strtoul",
    "strtoll",
}

_ALLOC_CALLS = {
    "malloc",
    "calloc",
    "realloc",
    "kmalloc",
    "kzalloc",
    "vmalloc",
    "g_malloc",
    "g_new",
    "g_new0",
}

_RETURN_CALLS = {
    "read",
    "write",
    "recv",
    "send",
    "fread",
    "fwrite",
    "open",
    "close",
    "fopen",
    "fclose",
}


@dataclass
class StaticAnalysisResult:
    risk_indicators: List[str] = field(default_factory=list)
    defense_indicators: List[str] = field(default_factory=list)
    defense_flags: Dict[str, bool] = field(
        default_factory=lambda: {name: False for name in DEFENSE_PATTERNS}
    )
    risk_score: float = 0.0
    defense_score: float = 0.0
    summary: str = ""
    source: str = "lexical"
    dangerous_calls: int = 0
    null_derefs_unguarded: int = 0
    array_access_unguarded: int = 0
    integer_overflow_risk: int = 0
    alloc_without_check: int = 0
    total_defense_patterns: int = 0
    total_null_checks: int = 0
    total_bounds_checks: int = 0
    total_error_returns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_indicators": list(self.risk_indicators),
            "defense_indicators": list(self.defense_indicators),
            "defense_flags": dict(self.defense_flags),
            "risk_score": round(self.risk_score, 3),
            "defense_score": round(self.defense_score, 3),
            "summary": self.summary,
            "source": self.source,
            "dangerous_calls": self.dangerous_calls,
            "null_derefs_unguarded": self.null_derefs_unguarded,
            "array_access_unguarded": self.array_access_unguarded,
            "integer_overflow_risk": self.integer_overflow_risk,
            "alloc_without_check": self.alloc_without_check,
            "total_defense_patterns": self.total_defense_patterns,
            "total_null_checks": self.total_null_checks,
            "total_bounds_checks": self.total_bounds_checks,
            "total_error_returns": self.total_error_returns,
        }

    def audit_flags(self) -> Dict[str, bool]:
        return {name: bool(self.defense_flags.get(name, False)) for name in DEFENSE_PATTERNS}


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _matches(pattern: str, text: str) -> Iterable[re.Match[str]]:
    return re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE)


def _has_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _parse_tree(code: str) -> bool:
    if _tree_sitter is None or _tree_sitter_c is None:
        return False
    try:
        language = _tree_sitter.Language(_tree_sitter_c.language())
        parser = _tree_sitter.Parser(language)
        tree = parser.parse(code.encode("utf-8", errors="replace"))
        return not tree.root_node.has_error
    except Exception:
        return False


def _scan_risks(code: str, result: StaticAnalysisResult) -> None:
    for match in _matches(r"\b([A-Za-z_]\w*)\s*\(", code):
        name = match.group(1)
        if name in _UNSAFE_CALLS:
            result.dangerous_calls += 1
            line = _line_number(code, match.start())
            _append_unique(
                result.risk_indicators,
                f"call {name} at line {line}: {_UNSAFE_CALLS[name]}",
            )

    for match in _matches(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]", code):
        index = match.group(2).strip()
        if not index.isdigit() and not re.fullmatch(r"[0-9+* ()\-/]+", index):
            line = _line_number(code, match.start())
            if not _has_any(code[max(0, match.start() - 240):match.start()], ("<", ">", "bound", "size", "len", "count")):
                result.array_access_unguarded += 1
                _append_unique(result.risk_indicators, f"variable index at line {line} lacks a nearby bound")

    dereferences = [(match.group(1), match.start()) for match in _matches(r"\b([A-Za-z_]\w*)\s*->", code)]
    for match in _matches(r"\*\s*([A-Za-z_]\w*)", code):
        prefix = code[max(0, match.start() - 32):match.start()]
        if re.search(r"\b(?:char|short|int|long|float|double|void|size_t|struct\s+\w+)\s*$", prefix, re.I):
            continue
        dereferences.append((match.group(1), match.start()))
    for name, offset in dereferences:
        prefix = code[max(0, offset - 220):offset]
        if not re.search(rf"(?:!|\b){re.escape(name)}\s*(?:==|!=)\s*(?:0|null|nullptr)", prefix, re.I):
            result.null_derefs_unguarded += 1
            line = _line_number(code, offset)
            _append_unique(result.risk_indicators, f"pointer {name} at line {line} lacks a visible null guard")

    for match in _matches(r"\b(?:malloc|calloc|realloc|kmalloc|kzalloc|vmalloc|g_malloc|g_new|g_new0)\s*\([^)]*[+*][^)]*\)", code):
        prefix = code[match.start():match.end() + 220]
        if not _has_any(prefix, ("overflow", "size_max", "checked", "builtin_mul", "builtin_add")):
            result.integer_overflow_risk += 1
            line = _line_number(code, match.start())
            _append_unique(result.risk_indicators, f"allocation arithmetic at line {line} lacks an overflow guard")

    for match in _matches(r"\b(?:malloc|calloc|realloc|kmalloc|kzalloc|vmalloc|g_malloc|g_new|g_new0)\s*\(", code):
        suffix = code[match.end():match.end() + 180]
        if not re.search(r"(?:==|!=|!|is_err|nonnull|assert)", suffix, re.I):
            result.alloc_without_check += 1

    if result.dangerous_calls:
        _append_unique(result.risk_indicators, f"{result.dangerous_calls} unbounded or size-sensitive call(s)")
    if result.null_derefs_unguarded:
        _append_unique(result.risk_indicators, f"{result.null_derefs_unguarded} possible unguarded pointer dereference(s)")
    if result.array_access_unguarded:
        _append_unique(result.risk_indicators, f"{result.array_access_unguarded} possible unguarded indexed access(es)")
    if result.integer_overflow_risk:
        _append_unique(result.risk_indicators, f"{result.integer_overflow_risk} allocation expression(s) with arithmetic")
    if result.alloc_without_check:
        _append_unique(result.risk_indicators, f"{result.alloc_without_check} allocation result(s) without a nearby check")


def _scan_defenses(code: str, result: StaticAnalysisResult) -> None:
    null_checks = list(_matches(r"(?:if\s*\([^)]*(?:null|nullptr|==\s*0|!=\s*0|!\s*[A-Za-z_]\w*)[^)]*\)|is_err(?:_or_null)?\s*\([^)]*\))", code))
    bound_checks = list(_matches(r"(?:if\s*\([^)]*\b(?:size|len|length|count|capacity|bound|limit|n|index|offset)\b[^)]*(?:<|>|<=|>=)[^)]*\)|assert\s*\([^)]*(?:size|len|length|count|n|index)[^)]*\))", code))
    return_checks = list(_matches(r"(?:if\s*\([^)]*(?:ret|rc|status|result|error|fail|<\s*0|==\s*-1|!=\s*0)[^)]*\)\s*(?:\{|return|goto))", code))
    input_checks = list(_matches(r"(?:validate|sanitize|allowlist|whitelist|parse|is_valid|check_input|range_check)\s*\(", code))
    safe_calls = list(_matches(r"\b(?:" + "|".join(map(re.escape, _SAFE_CALLS)) + r")\s*\(", code))
    overflow_checks = list(_matches(r"(?:builtin_(?:add|sub|mul)_overflow|size_max|uint_max|checked_(?:add|sub|mul)|safe_(?:add|mul)|overflow)", code))

    result.total_null_checks = len(null_checks)
    result.total_bounds_checks = len(bound_checks)
    result.total_error_returns = len(return_checks)
    result.total_defense_patterns = len(null_checks) + len(bound_checks) + len(return_checks) + len(input_checks) + len(safe_calls) + len(overflow_checks)

    result.defense_flags.update({
        "bounds": bool(bound_checks),
        "null": bool(null_checks),
        "return_value": bool(return_checks),
        "input_validation": bool(input_checks),
        "safe_api": bool(safe_calls),
        "error_handling": bool(return_checks) or _has_any(code, ("cleanup", "rollback", "finally", "goto error")),
        "overflow": bool(overflow_checks),
    })

    labels = {
        "bounds": "bounds or size guard",
        "null": "null guard",
        "return_value": "return-value check",
        "input_validation": "input validation",
        "safe_api": "bounded or checked API",
        "error_handling": "error-handling path",
        "overflow": "overflow guard",
    }
    for name, present in result.defense_flags.items():
        if present:
            _append_unique(result.defense_indicators, labels[name])


def _score(result: StaticAnalysisResult) -> None:
    risk = (
        0.14 * result.dangerous_calls
        + 0.10 * result.null_derefs_unguarded
        + 0.12 * result.array_access_unguarded
        + 0.15 * result.integer_overflow_risk
        + 0.08 * result.alloc_without_check
    )
    result.risk_score = min(1.0, risk)
    result.defense_score = min(1.0, 0.08 * sum(result.defense_flags.values()))


def _summary(result: StaticAnalysisResult) -> str:
    risks = "; ".join(result.risk_indicators[:8]) or "none"
    defenses = "; ".join(result.defense_indicators[:8]) or "none"
    return (
        f"Static evidence source={result.source}; risk={result.risk_score:.2f}; "
        f"defense={result.defense_score:.2f}. Risks: {risks}. Defenses: {defenses}."
    )


def analyze_code(code: str) -> StaticAnalysisResult:
    result = StaticAnalysisResult()
    if not code or not code.strip():
        result.summary = "No source was supplied."
        return result
    result.source = "tree-sitter" if _parse_tree(code) else "lexical"
    _scan_risks(code, result)
    _scan_defenses(code, result)
    _score(result)
    result.summary = _summary(result)
    return result


def get_prompt_section(analysis: StaticAnalysisResult) -> str:
    return analysis.summary


def merge_defense_flags(
    static_result: StaticAnalysisResult | None,
    llm_flags: Mapping[str, Any] | None,
) -> Dict[str, bool]:
    static_flags = static_result.audit_flags() if static_result is not None else {}
    source_flags = llm_flags or {}
    return {
        name: bool(static_flags.get(name, False)) or bool(source_flags.get(name, False))
        for name in DEFENSE_PATTERNS
    }
