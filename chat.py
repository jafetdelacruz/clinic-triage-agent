"""Talk to the agent in the terminal, as if you were a patient on WhatsApp.

Tool calls are printed in grey so you can see what the agent does behind each reply.
"""
import json
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from agent import new_conversation, run_turn  # noqa: E402  (needs .env loaded first)
from tools import Clinic  # noqa: E402
from tracing import Trace  # noqa: E402

GREY, RESET = "\033[90m", "\033[0m"


def main():
    client, clinic = OpenAI(), Clinic()
    trace = Trace(f"chat-{int(time.time())}")
    messages = new_conversation()
    print("Clinica Dental Sonrisa (type 'exit' to quit)\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": text})
        seen = sum(1 for _ in trace.path.open(encoding="utf-8"))
        reply = run_turn(client, clinic, messages, trace)
        for line in trace.path.read_text(encoding="utf-8").splitlines()[seen:]:
            event = json.loads(line)
            if event["event"] == "tool_call":
                print(f"{GREY}  [tool] {event['name']}({event['args']}) -> {event['result']}{RESET}")
        print(f"Agent: {reply}\n")
    print(f"{GREY}Trace: {trace.path} | {trace.total_tokens} tokens | {trace.total_ms} ms in model calls{RESET}")


if __name__ == "__main__":
    main()
