from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Sequence

from agent_router import TOP_LEVEL_CWES
from tools import llm_tool
from tools.static_analysis import DEFENSE_PATTERNS, StaticAnalysisResult, merge_defense_flags


ROLE_NAMES = ("RGR", "EGR", "XGR")
ROLE_ALIASES = {
    "rgr": "RGR",
    "egr": "EGR",
    "xgr": "XGR",
}

_BASE_WEIGHTS = {
    "bounds": 0.75,
    "null": 0.70,
    "return_value": 0.55,
    "input_validation": 0.55,
    "safe_api": 0.45,
    "error_handling": 0.50,
    "overflow": 0.70,
}

WEIGHT_MATRIX: Dict[str, Dict[str, float]] = {
    cwe: dict(_BASE_WEIGHTS) for cwe in TOP_LEVEL_CWES
}
for _cwe in ("CWE-787", "CWE-119"):
    WEIGHT_MATRIX[_cwe]["bounds"] = 1.0
for _cwe in ("CWE-476", "CWE-125"):
    WEIGHT_MATRIX[_cwe]["null"] = 1.0
for _cwe in ("CWE-190",):
    WEIGHT_MATRIX[_cwe]["overflow"] = 1.0
for _cwe in ("CWE-20", "CWE-22", "CWE-78", "CWE-89"):
    WEIGHT_MATRIX[_cwe]["input_validation"] = 1.0

THRESHOLDS = {cwe: 0.55 for cwe in TOP_LEVEL_CWES}


def _fallback(role: str, reason: str = "review unavailable") -> Dict[str, Any]:
    value = {
        "vuln": 0,
        "confidence": 0.3,
        "reason": reason,
        "evidence": "",
        "matched_pattern": "",
        "matched_cause": "",
        "role": role,
    }
    return value


def _json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("missing JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON result is not an object")
    return value


def _binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "vulnerable"}:
            return 1
        if normalised in {"0", "false", "no", "safe"}:
            return 0
    raise ValueError("verdict is not binary")


def _parse_review(text: str, role: str) -> Dict[str, Any]:
    value = _json_object(text)
    required = {"vuln", "confidence", "reason", "evidence"}
    if role == "EGR":
        required.update({"matched_pattern", "matched_cause"})
    missing = required.difference(value)
    if missing:
        raise ValueError("missing review fields")
    confidence = float(value["confidence"])
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence outside range")
    return {
        "vuln": _binary(value["vuln"]),
        "confidence": confidence,
        "reason": str(value.get("reason", ""))[:800],
        "evidence": str(value.get("evidence", ""))[:800],
        "matched_pattern": str(value.get("matched_pattern", ""))[:400],
        "matched_cause": str(value.get("matched_cause", ""))[:400],
        "role": role,
    }


def _call_review(prompt: str, role: str, model: str | None) -> Dict[str, Any]:
    for _ in range(2):
        try:
            return _parse_review(llm_tool.chat_completion(prompt, model=model, temperature=0.0), role)
        except Exception:
            pass
    return _fallback(role, f"{role} returned no valid structured review")


def _prompt_header(role: str) -> str:
    return (
        f"You are the {role} reviewer in a defense-anchored panel. "
        "Use only concrete code evidence. A vulnerable verdict requires a reachable "
        "operation and an absent defense. Cite one or more source line numbers in evidence."
    )


def _rgr_prompt(code: str, static_section: str) -> str:
    return (
        f"{_prompt_header('RGR')}\n"
        "Check memory, resource, integer, input, null, and error safety rules. "
        "For each candidate issue, inspect the same region for a neutralising guard.\n\n"
        f"Static evidence:\n{static_section}\n\nFunction:\n{code}\n\n"
        "Return JSON: {\"vuln\":0|1,\"confidence\":0..1,\"reason\":\"...\",\"evidence\":\"line ...\"}"
    )


def _egr_prompt(code: str, paired_section: str) -> str:
    return (
        f"{_prompt_header('EGR')}\n"
        "Compare the function with both sides of each paired reference. A vulnerable "
        "vote is allowed only when the vulnerable abstraction matches and the aligned "
        "patched mechanism is absent. Record the matched pattern and cause.\n\n"
        f"Paired references:\n{paired_section or 'none'}\n\nFunction:\n{code}\n\n"
        "Return JSON: {\"vuln\":0|1,\"confidence\":0..1,\"reason\":\"...\","
        "\"evidence\":\"line ...\",\"matched_pattern\":\"...\",\"matched_cause\":\"...\"}"
    )


def _xgr_prompt(code: str, static_section: str) -> str:
    return (
        f"{_prompt_header('XGR')}\n"
        "Hypothesise two or three adverse outcomes, trace the most threatening source "
        "to sink path, and self-critique every guard on that path. Retract to safe if "
        "a cited guard blocks the path.\n\n"
        f"Static evidence:\n{static_section}\n\nFunction:\n{code}\n\n"
        "Return JSON: {\"vuln\":0|1,\"confidence\":0..1,\"reason\":\"...\",\"evidence\":\"line ...\"}"
    )


def rgr_detect(code: str, sa_section: str = "", model: str | None = None, **_: Any) -> Dict[str, Any]:
    return _call_review(_rgr_prompt(code, sa_section), "RGR", model)


def egr_detect(code: str, few_shot_section: str = "", model: str | None = None, **_: Any) -> Dict[str, Any]:
    return _call_review(_egr_prompt(code, few_shot_section), "EGR", model)


def xgr_detect(code: str, sa_section: str = "", model: str | None = None, **_: Any) -> Dict[str, Any]:
    return _call_review(_xgr_prompt(code, sa_section), "XGR", model)


def _line_numbers(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"\bline(?:s)?\s*:?\s*(\d+)", text, flags=re.IGNORECASE)]


def _cited_line_exists(code: str, result: Mapping[str, Any]) -> bool:
    lines = code.splitlines()
    return any(1 <= number <= len(lines) for number in _line_numbers(str(result.get("evidence", "")) + " " + str(result.get("reason", ""))))


def _debate_prompt(code: str, role: str, own: Mapping[str, Any], peers: Mapping[str, Mapping[str, Any]]) -> str:
    peer_text = []
    for name, result in peers.items():
        if name == role:
            continue
        peer_text.append(
            f"{name}: verdict={'vulnerable' if result.get('vuln') else 'safe'}; "
            f"reason={str(result.get('reason', ''))[:500]}; evidence={str(result.get('evidence', ''))[:300]}"
        )
    peer_block = "\n".join(peer_text)
    schema = (
        "Return JSON with vuln, confidence, reason, evidence, matched_pattern, and matched_cause."
        if role == "EGR"
        else "Return JSON with vuln, confidence, reason, and evidence."
    )
    return (
        f"{_prompt_header(role)}\n"
        "Reconsider your prior verdict using the peer evidence below. Identify a new "
        "line-level fact, locate it in the function, and revise only when the cited "
        "line exists and supports the change. Otherwise hold your prior verdict.\n\n"
        f"Prior verdict={'vulnerable' if own.get('vuln') else 'safe'}\n"
        f"Prior reason={str(own.get('reason', ''))[:500]}\n"
        f"Peers:\n{peer_block}\n\nFunction:\n{code}\n\n"
    ) + schema


def _debate_round(code: str, current: Mapping[str, Mapping[str, Any]], model: str | None) -> Dict[str, Dict[str, Any]]:
    updated: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(current)) as pool:
        futures = {
            pool.submit(_call_review, _debate_prompt(code, role, current[role], current), role, model): role
            for role in current
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                candidate = future.result()
            except Exception:
                candidate = _fallback(role)
            previous = current[role]
            if int(candidate.get("vuln", 0)) != int(previous.get("vuln", 0)) and not _cited_line_exists(code, candidate):
                candidate = dict(previous)
            updated[role] = candidate
    return {role: updated.get(role, dict(current[role])) for role in current}


def _consensus(results: Mapping[str, Mapping[str, Any]]) -> bool:
    values = [int(result.get("vuln", 0)) for result in results.values()]
    return bool(values) and all(value == values[0] for value in values)


def _audit_prompt(code: str, rationale: str) -> str:
    names = ", ".join(DEFENSE_PATTERNS)
    return (
        "You are a fresh-context defense auditor. Inspect only the function and the "
        "final rationale. For each named defense pattern return true when a concrete "
        "guard is present near the claimed operation. Do not use peer deliberation.\n\n"
        f"Patterns: {names}\nFinal rationale:\n{rationale[:800]}\n\n"
        f"Function:\n{code}\n\nReturn JSON {{\"flags\":{{\"bounds\":true|false,...}}}}."
    )


def _parse_audit(text: str) -> Dict[str, bool]:
    value = _json_object(text)
    flags = value.get("flags", value)
    if not isinstance(flags, dict):
        raise ValueError("audit flags are not an object")
    return {name: bool(_binary(flags.get(name, False))) for name in DEFENSE_PATTERNS}


def asymmetric_defense_audit(
    code: str,
    rationale: str,
    cwe_id: str,
    static_result: StaticAnalysisResult | None = None,
    model: str | None = None,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    llm_flags = {name: False for name in DEFENSE_PATTERNS}
    for _ in range(2):
        try:
            llm_flags = _parse_audit(llm_tool.chat_completion(_audit_prompt(code, rationale), model=model, temperature=0.0))
            break
        except Exception:
            llm_flags = {name: False for name in DEFENSE_PATTERNS}
    fused = merge_defense_flags(static_result, llm_flags)
    matrix = weights or WEIGHT_MATRIX
    row = matrix.get(cwe_id, _BASE_WEIGHTS)
    denominator = sum(float(row.get(name, 0.0)) for name in DEFENSE_PATTERNS) or 1.0
    score = sum(float(row.get(name, 0.0)) * int(fused[name]) for name in DEFENSE_PATTERNS) / denominator
    threshold = float((thresholds or THRESHOLDS).get(cwe_id, 0.55))
    return {
        "llm_flags": llm_flags,
        "static_flags": static_result.audit_flags() if static_result is not None else {name: False for name in DEFENSE_PATTERNS},
        "fused_flags": fused,
        "score": round(score, 4),
        "threshold": threshold,
        "flip": bool(score >= threshold),
    }


def three_head_detect(
    code: str,
    sa_section: str = "",
    few_shot_section: str = "",
    max_debate_rounds: int = 2,
    model: str | None = None,
    skip_verify: bool = False,
    debate_method: str = "consensus",
    agents_to_use: Sequence[str] | None = None,
    static_result: StaticAnalysisResult | None = None,
    cwe_id: str = "CWE-UNKNOWN",
    audit_weights: Mapping[str, Mapping[str, float]] | None = None,
    audit_thresholds: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    requested = list(agents_to_use or ROLE_NAMES)
    roles = [ROLE_ALIASES.get(str(role).lower(), str(role).upper()) for role in requested]
    roles = [role for role in roles if role in ROLE_NAMES]
    if not roles:
        roles = list(ROLE_NAMES)
    prompts = {
        "RGR": lambda: _rgr_prompt(code, sa_section),
        "EGR": lambda: _egr_prompt(code, few_shot_section),
        "XGR": lambda: _xgr_prompt(code, sa_section),
    }
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        futures = {pool.submit(_call_review, prompts[role](), role, model): role for role in roles}
        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception:
                results[role] = _fallback(role)
    results = {role: results.get(role, _fallback(role)) for role in roles}

    rounds = 0
    if debate_method == "consensus":
        while not _consensus(results) and rounds < max(0, int(max_debate_rounds)):
            rounds += 1
            results = _debate_round(code, results, model)

    agreed = _consensus(results)
    if agreed:
        verdict = int(next(iter(results.values())).get("vuln", 0))
        confidence = sum(float(value.get("confidence", 0.3)) for value in results.values()) / len(results)
        final_reason = str(max(results.values(), key=lambda value: float(value.get("confidence", 0.0))).get("reason", ""))
    else:
        verdict = 0
        confidence = 0.40
        final_reason = "Unresolved reviewer panel; conservative safe fallback."

    best = max(results.values(), key=lambda value: float(value.get("confidence", 0.0)))
    audit: Dict[str, Any] | None = None
    if verdict == 1 and not skip_verify:
        audit = asymmetric_defense_audit(
            code,
            str(best.get("reason", "")),
            cwe_id,
            static_result,
            model=model,
            weights=audit_weights,
            thresholds=audit_thresholds,
        )
        if audit["flip"]:
            verdict = 0

    return {
        "vuln": verdict,
        "confidence": round(confidence, 4),
        "reason": final_reason[:800],
        "evidence": str(best.get("evidence", ""))[:800],
        "audit": audit,
        "debate_info": {
            "roles": roles,
            "rounds": rounds,
            "consensus": _consensus(results),
            "reviewers": results,
        },
    }


DIRECT_COT_PROMPT = (
    "Review this function for one concrete vulnerability and its defenses. "
    "Return JSON with vuln, confidence, reason, and evidence.\n\nFunction:\n{code}"
)


def direct_cot_detect(code: str, sa_section: str = "", few_shot_section: str = "", model: str | None = None) -> Dict[str, Any]:
    del sa_section, few_shot_section
    return _call_review(DIRECT_COT_PROMPT.format(code=code), "XGR", model)


def _parse_result(text: str) -> Dict[str, Any]:
    return _parse_review(text, "RGR")
