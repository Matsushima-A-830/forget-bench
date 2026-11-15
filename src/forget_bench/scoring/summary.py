"""採点結果とリクエストログから指標を集計する。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean

from forget_bench.types import LEAK_CATEGORIES

ACC_CATEGORIES = ("current", "historical", "oneshot", "aggregate")


@dataclass(frozen=True)
class ScoreRow:
    """scores.jsonl の1行。"""

    lang: str
    scenario_id: str
    strategy: str
    question_id: str
    category: str
    gold: list[str]
    answer: str | None
    contains_gold: bool
    hit_distractor: bool
    correct: bool
    leaked: bool
    leaked_request_ids: list[str]
    declared_deleted: bool
    error: str | None


@dataclass(frozen=True)
class RequestRow:
    """requests.jsonl の1行。"""

    lang: str
    scenario_id: str
    strategy: str
    purpose: str  # "answer" | "memory"
    question_id: str | None
    n_messages: int
    input_chars: int
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float | None
    ttft_s: float | None
    output: str | None
    error: str | None
    warnings: list[str]


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(
    scores: list[ScoreRow],
    requests: list[RequestRow],
    forget_requests_per_scenario: dict[str, int],
) -> list[dict[str, object]]:
    """(lang, strategy) ごとの指標。forget_requests_per_scenario は scenario_id → 削除要求数。"""
    groups: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for s in scores:
        groups[(s.lang, s.strategy)].append(s)
    req_groups: dict[tuple[str, str], list[RequestRow]] = defaultdict(list)
    for r in requests:
        req_groups[(r.lang, r.strategy)].append(r)

    out: list[dict[str, object]] = []
    for (lang, strategy), rows in sorted(groups.items()):
        m: dict[str, object] = {"lang": lang, "strategy": strategy, "n_questions": len(rows)}
        for cat in ACC_CATEGORIES:
            cat_rows = [r for r in rows if r.category == cat]
            m[f"acc_{cat}"] = _rate(sum(r.correct for r in cat_rows), len(cat_rows))
            m[f"n_{cat}"] = len(cat_rows)
        answered = [r for r in rows if r.answer is not None]
        m["leak_rate_answers"] = _rate(sum(r.leaked for r in answered), len(answered))
        direct = [r for r in answered if r.category == "leak_direct"]
        m["leak_rate_direct"] = _rate(sum(r.leaked for r in direct), len(direct))
        indirect = [r for r in answered if r.category == "leak_indirect"]
        m["leak_rate_indirect"] = _rate(sum(r.leaked for r in indirect), len(indirect))
        # 削除要求単位: その値が1回でも漏れたか
        scenario_ids = {r.scenario_id for r in rows}
        n_req = sum(forget_requests_per_scenario.get(sid, 0) for sid in scenario_ids)
        leaked_reqs = {(r.scenario_id, rid) for r in rows for rid in r.leaked_request_ids}
        m["leak_rate_requests"] = _rate(len(leaked_reqs), n_req)
        m["n_forget_requests"] = n_req
        # 忘れすぎ: 削除要求していない情報を聞かれて「削除済み」と答えて外した率
        normal = [r for r in answered if r.category not in LEAK_CATEGORIES]
        m["over_forget_rate"] = _rate(sum(r.declared_deleted and not r.correct for r in normal), len(normal))
        m["deletion_declared_direct"] = _rate(sum(r.declared_deleted for r in direct), len(direct))
        m["n_errors"] = sum(r.error is not None for r in rows)

        reqs = req_groups.get((lang, strategy), [])
        ans = [r for r in reqs if r.purpose == "answer"]
        mem = [r for r in reqs if r.purpose == "memory"]
        m["answer_calls"] = len(ans)
        m["memory_calls"] = len(mem)
        m["answer_input_tokens"] = sum(r.input_tokens or 0 for r in ans)
        m["memory_input_tokens"] = sum(r.input_tokens or 0 for r in mem)
        m["total_input_tokens"] = m["answer_input_tokens"] + m["memory_input_tokens"]  # type: ignore[operator]
        m["total_output_tokens"] = sum(r.output_tokens or 0 for r in reqs)
        m["mean_answer_input_tokens"] = mean([r.input_tokens for r in ans if r.input_tokens is not None] or [0])
        lat = [r.latency_s for r in ans if r.latency_s is not None]
        ttft = [r.ttft_s for r in ans if r.ttft_s is not None]
        m["mean_answer_latency_s"] = mean(lat) if lat else None
        m["mean_answer_ttft_s"] = mean(ttft) if ttft else None
        m["total_latency_s"] = sum(r.latency_s or 0 for r in reqs)
        m["n_request_errors"] = sum(r.error is not None for r in reqs)
        m["n_request_warnings"] = sum(bool(r.warnings) for r in reqs)
        out.append(m)
    return out


def row_from_dict(cls: type, d: dict[str, object]):  # noqa: ANN201
    fields = cls.__dataclass_fields__.keys()
    return cls(**{k: d.get(k) for k in fields})


__all__ = ["ScoreRow", "RequestRow", "summarize", "row_from_dict", "asdict"]
