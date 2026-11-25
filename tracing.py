"""Observability: every model call and tool call goes to a JSONL file.

One line per event, so a conversation can be replayed and audited later:
what the model saw, which tool it chose, with which arguments, what came
back, how long it took and how many tokens it cost.
"""
import json
import time
from pathlib import Path


class Trace:
    def __init__(self, conversation_id, folder="traces"):
        Path(folder).mkdir(exist_ok=True)
        self.path = Path(folder) / f"{conversation_id}.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.total_tokens = 0
        self.total_ms = 0

    def log(self, event, **data):
        if event == "llm_call":
            self.total_tokens += data.get("tokens", 0)
            self.total_ms += data.get("latency_ms", 0)
        line = {"ts": round(time.time(), 3), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
