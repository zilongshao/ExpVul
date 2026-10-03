from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import agent_router
from agents.reasoning_agents import three_head_detect
from tools import llm_tool
from tools.static_analysis import analyze_code, get_prompt_section
from tools.vulnerability_pattern_index import VulnerabilityPatternIndex


@lru_cache(maxsize=4)
def _index(patterns_path: str | None, cache_path: str | None) -> VulnerabilityPatternIndex | None:
    if not patterns_path and not cache_path:
        return None
    value = VulnerabilityPatternIndex(patterns_path=patterns_path, cache_path=cache_path)
    value.load()
    return value if value.patterns else None


def _selected_indices(text: str, count: int, limit: int) -> List[int]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        values = json.loads(text[start:end + 1])
    except (ValueError, TypeError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    selected: List[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < limit and index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def _paired_section(code: str, cwe_id: str, index: VulnerabilityPatternIndex | None, model: str | None) -> str:
    if index is None:
        return ""
    candidates = index.search(code, top_k=20)
    if not candidates:
        return ""
    selected = candidates[:2]
    if len(candidates) > 2 and llm_tool.configured():
        prompt = (
            "Select the two paired references most relevant to the function. "
            "Return only a JSON array of two candidate indices.\n\n"
            f"Routed weakness: {cwe_id}\nFunction:\n{code}\n\nCandidates:\n"
            f"{index.format_for_selection(candidates)}"
        )
        try:
            indices = _selected_indices(
                llm_tool.chat_completion(prompt, model=model, temperature=0.0),
                2,
                len(candidates),
            )
            chosen = index.get_patterns_by_indices(candidates, indices)
            if len(chosen) == 2:
                selected = chosen
        except Exception:
            selected = candidates[:2]
    return index.format_for_prompt(selected)


def detect_single(
    code: str,
    *,
    patterns_path: str | None = None,
    cache_path: str | None = None,
    model: str | None = None,
    routing_strategy: str = "full",
    max_debate_rounds: int = 2,
    skip_audit: bool = False,
) -> Dict[str, Any]:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("a non-empty function is required")
    static_result = analyze_code(code)
    routed = agent_router.detect_possible_cwes(code, mode=routing_strategy, model=model)
    primary = routed[0] if routed else "CWE-UNKNOWN"
    index = _index(patterns_path, cache_path)
    result = three_head_detect(
        code,
        sa_section=get_prompt_section(static_result),
        few_shot_section=_paired_section(code, primary, index, model),
        max_debate_rounds=max_debate_rounds,
        model=model,
        skip_verify=skip_audit,
        static_result=static_result,
        cwe_id=primary,
    )
    return {
        "vuln": int(result.get("vuln", 0)),
        "confidence": round(float(result.get("confidence", 0.3)), 4),
        "reason": str(result.get("reason", ""))[:800],
        "evidence": str(result.get("evidence", ""))[:800],
        "cwe": routed,
        "static_analysis": static_result.to_dict(),
        "audit": result.get("audit"),
        "review": result.get("debate_info", {}),
        "model_configured": llm_tool.configured(),
    }


def _record_code(record: Mapping[str, Any]) -> str:
    for key in ("code", "function", "source"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("record has no function text")


def _output_record(record: Mapping[str, Any], result: Mapping[str, Any], elapsed: float) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "id": record.get("id"),
        "result": dict(result),
        "elapsed_seconds": round(elapsed, 3),
    }
    return value


def run_records(
    input_path: str,
    output_path: str | None,
    *,
    patterns_path: str | None = None,
    cache_path: str | None = None,
    model: str | None = None,
    routing_strategy: str = "full",
    max_debate_rounds: int = 2,
    skip_audit: bool = False,
) -> List[Dict[str, Any]]:
    path = Path(input_path).expanduser()
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                value = json.loads(raw)
                if isinstance(value, dict):
                    records.append(value)
    outputs: List[Dict[str, Any]] = []
    for record in records:
        started = time.monotonic()
        try:
            result = detect_single(
                _record_code(record),
                patterns_path=patterns_path,
                cache_path=cache_path,
                model=model,
                routing_strategy=routing_strategy,
                max_debate_rounds=max_debate_rounds,
                skip_audit=skip_audit,
            )
            outputs.append(_output_record(record, result, time.monotonic() - started))
        except Exception as exc:
            outputs.append({"id": record.get("id"), "error": type(exc).__name__})
    encoded = "\n".join(json.dumps(value, ensure_ascii=False) for value in outputs)
    if output_path:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + ("\n" if encoded else ""), encoding="utf-8")
    else:
        sys.stdout.write(encoded + ("\n" if encoded else ""))
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Defense-evidence function review")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--code", help="single function text")
    source.add_argument("--input", help="newline-delimited records with a code field")
    parser.add_argument("--output", help="optional newline-delimited result path")
    parser.add_argument("--patterns", help="frozen paired-pattern store")
    parser.add_argument("--cache", help="frozen embedding cache")
    parser.add_argument("--model", help="single backbone name")
    parser.add_argument(
        "--routing-strategy",
        choices=("full", "no-router", "llm-only", "heuristic-only"),
        default="full",
    )
    parser.add_argument("--max-debate-rounds", type=int, default=2)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.model:
        llm_tool.set_model(args.model)
    if not args.offline and not llm_tool.configured():
        parser.error("model endpoint, authentication, and backbone settings are required")
    if not args.offline and not args.cache:
        parser.error("a frozen paired embedding cache is required")
    frozen_index = _index(args.patterns, args.cache)
    if not args.offline and frozen_index is None:
        parser.error("the frozen paired-pattern artifact could not be loaded")
    if not args.offline and frozen_index.embeddings is None:
        parser.error("the frozen paired-pattern artifact has no embeddings")
    if not args.offline and frozen_index.faiss_index is None:
        parser.error("the frozen paired-pattern artifact has no FAISS index")
    if args.code is not None:
        result = detect_single(
            args.code,
            patterns_path=args.patterns,
            cache_path=args.cache,
            model=args.model,
            routing_strategy=args.routing_strategy,
            max_debate_rounds=args.max_debate_rounds,
            skip_audit=args.skip_audit,
        )
        encoded = json.dumps(result, ensure_ascii=False)
        if args.output:
            destination = Path(args.output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded + "\n", encoding="utf-8")
        else:
            sys.stdout.write(encoded + "\n")
        return 0
    run_records(
        args.input,
        args.output,
        patterns_path=args.patterns,
        cache_path=args.cache,
        model=args.model,
        routing_strategy=args.routing_strategy,
        max_debate_rounds=args.max_debate_rounds,
        skip_audit=args.skip_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
