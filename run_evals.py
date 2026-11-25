"""Run every case in evals/cases.jsonl against the real model and score it.

A case passes when the agent's *actions* match what we expect:
- it called the tools it had to call,
- it handed off to a person when it should (and not when it shouldn't),
- it created exactly the expected number of bookings.

We grade actions, not wording: the reply text can vary between runs,
but booking without a name or not escalating an emergency is a real bug.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI

load_dotenv()

from agent import MODEL, new_conversation, run_turn  # noqa: E402
from tools import Clinic  # noqa: E402
from tracing import Trace  # noqa: E402


def tools_called(trace):
    lines = trace.path.read_text(encoding="utf-8").splitlines()
    return [e["name"] for e in map(json.loads, lines) if e["event"] == "tool_call"]


def grade(case, called, clinic, early_bookings):
    problems = []
    if early_bookings:
        problems.append("booked before the last message")
    if case.get("name_contains") and not any(case["name_contains"] in b["patient_name"] for b in clinic.bookings):
        problems.append(f"no booking for {case['name_contains']}")
    for name in case["must_call"]:
        if name not in called:
            problems.append(f"did not call {name}")
    if case["handoff"] is not None and bool(clinic.handoffs) != case["handoff"]:
        problems.append("missed handoff" if case["handoff"] else "unneeded handoff")
    if len(clinic.bookings) != case["bookings"]:
        problems.append(f"{len(clinic.bookings)} bookings, expected {case['bookings']}")
    return problems


def main():
    client = OpenAI()
    cases = [json.loads(line) for line in Path("evals/cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results, tokens = [], 0
    for case in cases:
        clinic, trace, messages = Clinic(), Trace(f"eval-{case['id']}"), new_conversation()
        replies, early_bookings = [], 0
        try:
            for i, text in enumerate(case["turns"]):
                messages.append({"role": "user", "content": text})
                replies.append(run_turn(client, clinic, messages, trace))
                if i < len(case["turns"]) - 1:
                    early_bookings = len(clinic.bookings)
        except APIError as exc:  # one failing call should not abort the whole run
            replies.append(f"API error: {exc}")
            early_bookings = -1
        called = tools_called(trace)
        problems = grade(case, called, clinic, early_bookings)
        if early_bookings == -1:
            problems = ["API error"]
        tokens += trace.total_tokens
        results.append({"id": case["id"], "passed": not problems, "problems": problems, "tools": called, "last_reply": replies[-1] if replies else ""})
        print(f"{'PASS' if not problems else 'FAIL'}  {case['id']:<22} tools={called} {'; '.join(problems)}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed | model {MODEL} | {tokens} tokens")
    Path("evals/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
