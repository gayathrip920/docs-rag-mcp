"""Measure how good the written answers are, using your own OpenRouter key.

Run from the repo root (needs DATABASE_URL, the model libraries, and a key):

    OPENROUTER_API_KEY=... python -m evals.run_answer_eval
    python -m evals.run_answer_eval --limit 5      # a cheap trial run first

It asks every test question the same way the app does, then checks:
  - in-scope questions: was an answer given, are the citations real, and does
    the answer cite the page that holds the correct answer?
  - out-of-scope questions: did it say "Not found in the docs."?

It stops when the total cost reaches EVAL_MAX_COST_USD (default 2.00).
Results go to evals/results/answers.json. The key is never written anywhere.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import mean, percentile
from rag import answer, llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_COST_USD = 2.0
# Errors that will repeat for every question, so there is no point continuing.
FATAL_REASONS = {"missing_key", "odd_key", "bad_key", "no_credits", "bad_request"}


@dataclass
class Outcome:
    result: answer.AnswerResult
    context_pages: list[str]  # pages of the chunks the model was shown
    model_ms: float


def grade_in_scope(outcome: Outcome, gold_pages: list[str]) -> dict:
    gold = set(gold_pages)
    cited = {s["page_path"] for s in outcome.result.sources if s["cited"]}
    return {
        "found": outcome.result.found,
        "citations_valid": outcome.result.citations_valid,
        "gold_in_context": bool(gold & set(outcome.context_pages)),
        "cites_gold": bool(gold & cited),
    }


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def summarize(in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    """Turn per-question grades into the numbers shown in the README."""
    answered = [r for r in in_scope if r["found"]]
    with_context = [r for r in in_scope if r["gold_in_context"]]
    return {
        "in_scope_questions": len(in_scope),
        "answered_rate": _rate(len(answered), len(in_scope)),
        "valid_citations_rate": _rate(
            sum(1 for r in answered if r["citations_valid"]), len(answered)
        ),
        "cites_correct_page_rate": _rate(sum(r["cites_gold"] for r in in_scope), len(in_scope)),
        "cites_correct_page_when_it_was_retrieved": _rate(
            sum(r["cites_gold"] for r in with_context), len(with_context)
        ),
        "out_of_scope_questions": len(out_of_scope),
        "correct_refusal_rate": _rate(
            sum(1 for r in out_of_scope if not r["found"]), len(out_of_scope)
        ),
    }


def evaluate(
    in_scope_questions: list[dict],
    out_of_scope_questions: list[dict],
    ask,
    max_cost_usd: float,
    limit: int | None = None,
) -> dict:
    """Run every question through `ask(question) -> Outcome` and collect grades.

    Stops early if the cost cap is reached, and aborts on an error that would repeat.
    """
    in_scope_questions = in_scope_questions[:limit] if limit else in_scope_questions
    out_of_scope_questions = out_of_scope_questions[:limit] if limit else out_of_scope_questions
    state = {"cost": 0.0, "cost_missing": 0, "errors": [], "status": "complete", "times": []}
    graded = {"in": [], "out": []}

    def run(questions, bucket, grade):
        for item in questions:
            if state["cost"] >= max_cost_usd:
                state["status"] = "stopped_at_cost_cap"
                return False
            try:
                outcome = ask(item["question"])
            except llm.LLMError as exc:
                state["errors"].append({"id": item["id"], "reason": exc.reason})
                if exc.reason in FATAL_REASONS:
                    state["status"] = f"aborted_{exc.reason}"
                    return False
                continue
            cost = outcome.result.cost_usd
            if cost is None:
                state["cost_missing"] += 1
            else:
                state["cost"] += cost
            state["times"].append(outcome.model_ms)
            row = {"id": item["id"], "question": item["question"], **grade(outcome, item)}
            row["answer"] = outcome.result.answer[:600]
            graded[bucket].append(row)
        return True

    if run(in_scope_questions, "in", lambda o, item: grade_in_scope(o, item["gold_pages"])):
        run(out_of_scope_questions, "out", lambda o, item: {"found": o.result.found})

    return {
        "status": state["status"],
        "summary": summarize(graded["in"], graded["out"]),
        "cost_usd_total": round(state["cost"], 5),
        "cost_usd_per_question": round(state["cost"] / max(len(state["times"]), 1), 5),
        "answers_without_cost_info": state["cost_missing"],
        "model_ms_p50": round(percentile(state["times"], 50), 1),
        "model_ms_p95": round(percentile(state["times"], 95), 1),
        "mean_model_ms": round(mean(state["times"]), 1),
        "errors": state["errors"],
        "in_scope": graded["in"],
        "out_of_scope": graded["out"],
    }


def load_jsonl(name: str) -> list[dict]:
    lines = (ROOT / "evals" / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    from rag.config import load_settings
    from rag.db import connect
    from rag.retrieve import retrieve

    parser = argparse.ArgumentParser(description="Score the written answers.")
    parser.add_argument("--limit", type=int, help="only run this many questions of each kind")
    parser.add_argument("--mode", default="hybrid_rerank", help="search method used for retrieval")
    args = parser.parse_args()

    settings = load_settings()
    api_key = os.environ.get("OPENROUTER_API_KEY") or settings.openrouter_api_key
    max_cost = float(os.environ.get("EVAL_MAX_COST_USD", DEFAULT_MAX_COST_USD))
    model = settings.openrouter_model_eval
    try:
        llm.clean_key(api_key)  # fail now, before any searching
    except llm.LLMError as exc:
        print(f"{llm.public_message(exc)} Set OPENROUTER_API_KEY and run again.")
        return 2

    conn = connect()

    def ask(question: str) -> Outcome:
        hits = retrieve(conn, question, mode=args.mode, limit=5)
        if not hits:
            return Outcome(answer.finalize(answer.NOT_FOUND, []), [], 0.0)
        started = time.perf_counter()
        reply = llm.chat(answer.build_messages(question, hits), model, api_key)
        elapsed = (time.perf_counter() - started) * 1000
        return Outcome(
            answer.finalize(reply.text, hits, reply), [h.page_path for h in hits], elapsed
        )

    in_scope = load_jsonl("questions.jsonl")
    out_of_scope = load_jsonl("out_of_scope.jsonl")
    print(f"model {model}, cost cap ${max_cost:.2f}, search mode {args.mode}")
    report = evaluate(in_scope, out_of_scope, ask, max_cost, args.limit)

    output = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": model,
        "search_mode": args.mode,
        "cost_cap_usd": max_cost,
        **report,
        "note": (
            "cites_correct_page_rate is the share of in-scope questions where the answer cites "
            "the page that holds the correct answer. The 'when_it_was_retrieved' version only "
            "counts questions where that page was among the 5 chunks the model saw."
        ),
    }
    out_path = ROOT / "evals" / "results" / "answers.json"
    saved = not report["status"].startswith("aborted")  # an aborted run has nothing worth keeping
    if saved:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"status: {report['status']}")
    for name, value in report["summary"].items():
        print(f"  {name}: {value}")
    print(
        f"  cost: ${report['cost_usd_total']} in total, "
        f"${report['cost_usd_per_question']} per question"
    )
    print(f"saved {out_path.relative_to(ROOT)}" if saved else "run aborted, nothing saved")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
