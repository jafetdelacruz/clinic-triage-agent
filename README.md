# clinic-triage-agent

A small WhatsApp-style agent for a dental clinic, in Python. A patient writes; the agent decides whether to answer, look up a price, check the calendar, book an appointment or hand the chat to a person.

It is a stripped-down version of what I run in production at [Relevo.chat](https://relevo.chat), rebuilt in Python to show the parts that matter: tool calling, guardrails, tracing and evals.

## How it works

```
patient message
      |
      v
  agent.py  -- sends conversation + tool list --> LLM (OpenAI)
      ^                                              |
      |  tool result (JSON)          tool call?  <---+
      |                                 |
  tools.py  <---------------------------+   no tool call -> reply to patient
  (Clinic: prices, free slots, bookings, handoffs)

  every model call and tool call -> traces/<conversation>.jsonl
```

- `tools.py`: what the agent is allowed to do. Prices and slots come from `clinic_data.json`, never from the model. Booking removes the slot, so the same time can't be booked twice.
- `agent.py`: the loop. Max 6 steps per message; if the model keeps calling tools, the chat goes to a person.
- `tracing.py`: one JSONL line per event with arguments, results, latency and tokens.
- `run_evals.py` + `evals/cases.jsonl`: 12 scenarios graded on actions, not wording (did it book without a name? did it escalate the toothache?).
- `test_agent.py`: offline tests with a fake model, no API key needed.

## Run it

```bash
uv sync
cp .env.example .env   # add your OPENAI_API_KEY
uv run python -m unittest      # offline tests
uv run python chat.py          # talk to the agent
uv run python run_evals.py     # score it against the 12 cases
```

## What I'd add next

- A unique constraint on (date, time) in a real database: removing the slot from a list only works in one process.
- An LLM-as-judge pass for tone and language, on top of the action checks.
- Langfuse or similar instead of local JSONL once there is more than one user.
