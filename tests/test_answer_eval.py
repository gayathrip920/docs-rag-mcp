import json

from evals import run_answer_eval as ev
from rag import answer, llm
from rag.retrieve import Hit

SECRET = "sk-or-eval-secret"


def hits():
    return [
        Hit(1, "tutorial/cors.md", "CORS", "https://x/cors/", "cors text"),
        Hit(2, "tutorial/other.md", "Other", "https://x/other/", "other text"),
    ]


def outcome(text, cost=0.01, context=("tutorial/cors.md", "tutorial/other.md")):
    reply = llm.LLMResult(text, "m", 10, 5, cost, 20)
    return ev.Outcome(answer.finalize(text, hits(), reply), list(context), 20.0)


IN_SCOPE = [
    {"id": f"q{i}", "question": f"question {i}", "gold_pages": ["tutorial/cors.md"]}
    for i in range(1, 5)
]
OUT_OF_SCOPE = [{"id": f"o{i}", "question": f"outside {i}"} for i in range(1, 3)]


def test_grade_marks_an_answer_that_cites_the_right_page():
    graded = ev.grade_in_scope(outcome("Use the middleware [1]."), ["tutorial/cors.md"])
    assert graded == {
        "found": True,
        "citations_valid": True,
        "gold_in_context": True,
        "cites_gold": True,
    }


def test_grade_separates_a_retrieval_miss_from_a_wrong_citation():
    wrong_page = ev.grade_in_scope(outcome("It is here [2]."), ["tutorial/cors.md"])
    assert wrong_page["gold_in_context"] and not wrong_page["cites_gold"]

    not_retrieved = ev.grade_in_scope(
        outcome("It is here [2].", context=["tutorial/other.md"]), ["tutorial/cors.md"]
    )
    assert not not_retrieved["gold_in_context"]


def test_grade_flags_made_up_citations():
    graded = ev.grade_in_scope(outcome("It is here [1] and [9]."), ["tutorial/cors.md"])
    assert graded["citations_valid"] is False
    assert graded["cites_gold"] is True


def test_full_run_computes_rates_and_cost():
    def ask(question):
        if question.startswith("outside"):
            return outcome(answer.NOT_FOUND)
        return outcome("Use it [1]." if question != "question 4" else "Not found in the docs.")

    report = ev.evaluate(IN_SCOPE, OUT_OF_SCOPE, ask, max_cost_usd=1.0)
    summary = report["summary"]
    assert report["status"] == "complete"
    assert summary["answered_rate"] == 0.75
    assert summary["valid_citations_rate"] == 1.0
    assert summary["cites_correct_page_rate"] == 0.75
    assert summary["correct_refusal_rate"] == 1.0
    assert report["cost_usd_total"] == 0.06  # six questions at 0.01 each


def test_a_wrongly_answered_out_of_scope_question_lowers_the_refusal_rate():
    report = ev.evaluate([], OUT_OF_SCOPE, lambda q: outcome("Sure [1]."), max_cost_usd=1.0)
    assert report["summary"]["correct_refusal_rate"] == 0.0


def test_stops_when_the_cost_cap_is_reached():
    calls = []

    def ask(question):
        calls.append(question)
        return outcome("Use it [1].", cost=0.4)

    report = ev.evaluate(IN_SCOPE, OUT_OF_SCOPE, ask, max_cost_usd=0.5)
    assert report["status"] == "stopped_at_cost_cap"
    assert len(calls) == 2  # 0.4, then 0.8 which is over the cap
    assert report["summary"]["in_scope_questions"] == 2


def test_a_key_problem_aborts_the_run_at_once():
    def ask(question):
        raise llm.LLMError("bad_key")

    report = ev.evaluate(IN_SCOPE, OUT_OF_SCOPE, ask, max_cost_usd=1.0)
    assert report["status"] == "aborted_bad_key"
    assert report["errors"] == [{"id": "q1", "reason": "bad_key"}]


def test_a_one_off_error_is_recorded_and_the_run_continues():
    state = {"n": 0}

    def ask(question):
        state["n"] += 1
        if state["n"] == 2:
            raise llm.LLMError("rate_limited")
        return outcome("Use it [1].")

    report = ev.evaluate(IN_SCOPE, [], ask, max_cost_usd=1.0)
    assert report["status"] == "complete"
    assert report["errors"] == [{"id": "q2", "reason": "rate_limited"}]
    assert report["summary"]["in_scope_questions"] == 3


def test_missing_cost_is_counted_not_treated_as_a_crash():
    report = ev.evaluate(IN_SCOPE[:2], [], lambda q: outcome("Use it [1].", cost=None), 1.0)
    assert report["answers_without_cost_info"] == 2
    assert report["cost_usd_total"] == 0


def test_limit_trims_both_question_lists():
    report = ev.evaluate(IN_SCOPE, OUT_OF_SCOPE, lambda q: outcome("Use it [1]."), 1.0, limit=1)
    assert report["summary"]["in_scope_questions"] == 1
    assert report["summary"]["out_of_scope_questions"] == 1


def test_the_report_never_contains_the_key():
    def ask(question):
        raise llm.LLMError("rate_limited")

    report = ev.evaluate(IN_SCOPE, OUT_OF_SCOPE, ask, max_cost_usd=1.0)
    assert SECRET not in json.dumps(report)
