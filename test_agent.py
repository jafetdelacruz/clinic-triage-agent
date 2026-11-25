"""Offline tests: no API key needed. A fake model replays scripted answers."""
import json
import tempfile
import unittest
from types import SimpleNamespace as NS

from agent import FALLBACK, MAX_STEPS, new_conversation, run_turn
from tools import Clinic, run_tool
from tracing import Trace


def tool_call(name, args, call_id="c1"):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    msg = NS(content=content, tool_calls=calls)
    return NS(choices=[NS(message=msg)], usage=NS(total_tokens=10))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.chat = NS(completions=NS(create=lambda **kw: self.script.pop(0)))


class ToolTests(unittest.TestCase):
    def test_double_booking_is_blocked(self):
        clinic = Clinic()
        self.assertTrue(clinic.book_appointment("Ana", "Dental cleaning", "2026-12-02", "10:00")["ok"])
        second = clinic.book_appointment("Luis", "Dental cleaning", "2026-12-02", "10:00")
        self.assertFalse(second["ok"])
        self.assertEqual(len(clinic.bookings), 1)

    def test_price_lookup_is_fuzzy(self):
        self.assertEqual(Clinic().get_price("cleaning")["price_mxn"], 600)
        self.assertIn("error", Clinic().get_price("nose job"))

    def test_unknown_tool_and_bad_args(self):
        self.assertIn("error", run_tool(Clinic(), "delete_everything", {}))
        self.assertIn("error", run_tool(Clinic(), "get_price", {"wrong": 1}))


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def test_tool_then_answer(self):
        client = FakeClient([
            reply(calls=[tool_call("get_price", {"service": "cleaning"})]),
            reply(content="La limpieza cuesta $600."),
        ])
        clinic, messages = Clinic(), new_conversation() + [{"role": "user", "content": "precio limpieza"}]
        answer = run_turn(client, clinic, messages, Trace("t1", self.folder))
        self.assertEqual(answer, "La limpieza cuesta $600.")
        tool_msg = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(json.loads(tool_msg["content"])["price_mxn"], 600)

    def test_step_limit_hands_off(self):
        loop = [reply(calls=[tool_call("check_availability", {"date": "2026-12-01"})]) for _ in range(MAX_STEPS)]
        clinic = Clinic()
        answer = run_turn(FakeClient(loop), clinic, new_conversation(), Trace("t2", self.folder))
        self.assertEqual(answer, FALLBACK)
        self.assertEqual(len(clinic.handoffs), 1)


if __name__ == "__main__":
    unittest.main()
