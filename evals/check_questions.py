"""Sanity check for the question files.

Run from the repo root:  python evals/check_questions.py
It makes sure every question has an id, no ids repeat, and every gold page
really exists in data/raw.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def load(name):
    path = ROOT / "evals" / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main():
    problems = []
    questions = load("questions.jsonl")
    out_of_scope = load("out_of_scope.jsonl")

    ids = [row["id"] for row in questions + out_of_scope]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids found")

    for row in questions:
        if not row.get("question", "").strip():
            problems.append(f"{row['id']}: empty question")
        if not row.get("gold_pages"):
            problems.append(f"{row['id']}: no gold pages")
        for page in row.get("gold_pages", []):
            if not (RAW / page).is_file():
                problems.append(f"{row['id']}: gold page not found: {page}")

    for row in out_of_scope:
        if not row.get("question", "").strip():
            problems.append(f"{row['id']}: empty question")

    print(f"{len(questions)} in-scope questions, {len(out_of_scope)} out-of-scope questions")
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
