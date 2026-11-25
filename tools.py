"""The clinic's tools: what the agent is allowed to do in the real world.

The LLM never touches data directly. It asks for a tool by name, and this
module decides what actually happens. That keeps prices and schedules
honest: the model can't invent them.
"""
import json
from pathlib import Path

DATA = json.loads((Path(__file__).parent / "clinic_data.json").read_text(encoding="utf-8"))


class Clinic:
    """In-memory clinic. A fresh Clinic per conversation keeps evals independent."""

    def __init__(self, data=DATA):
        self.prices = dict(data["prices_mxn"])
        self.free_slots = {day: list(times) for day, times in data["free_slots"].items()}
        self.bookings = []
        self.handoffs = []

    def get_price(self, service):
        wanted = service.lower().strip()
        for name, price in self.prices.items():
            if wanted in name.lower() or name.lower() in wanted:
                return {"service": name, "price_mxn": price}
        return {"error": "Unknown service", "known_services": list(self.prices)}

    def check_availability(self, date):
        return {"date": date, "free_times": self.free_slots.get(date, [])}

    def book_appointment(self, patient_name, service, date, time):
        free = self.free_slots.get(date, [])
        if time not in free:
            return {"ok": False, "error": "Slot not available", "free_times": free}
        # Taking the slot out of the list is what prevents double booking.
        # ponytail: fine for one process; a real system needs a unique
        # constraint on (date, time) in the database, because two requests
        # can arrive at the same moment.
        free.remove(time)
        booking = {"patient_name": patient_name, "service": service, "date": date, "time": time}
        self.bookings.append(booking)
        return {"ok": True, "booking": booking}

    def handoff_to_human(self, reason, summary):
        self.handoffs.append({"reason": reason, "summary": summary})
        return {"ok": True, "note": "A person from the clinic will take over this chat."}


# What the model sees: name, description and JSON schema of each tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Look up the price of a dental service in MXN.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string", "description": "Service name, e.g. 'cleaning'"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "List free appointment times for one day.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "Day in YYYY-MM-DD format"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment. Only call it once the patient gave their name and confirmed service, day and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "service": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM, 24h"},
                },
                "required": ["patient_name", "service", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff_to_human",
            "description": "Pass the chat to a person: pain or emergencies, complaints, billing disputes, medical questions, cancellations, or when unsure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["emergency", "complaint", "billing", "medical", "cancellation", "unsure"]},
                    "summary": {"type": "string", "description": "One line for the staff member"},
                },
                "required": ["reason", "summary"],
            },
        },
    },
]

ALLOWED = {tool["function"]["name"] for tool in TOOLS}


def run_tool(clinic, name, args):
    """Run a tool the model asked for. Unknown names and bad arguments become errors the model can read."""
    if name not in ALLOWED:
        return {"error": f"Unknown tool {name}"}
    try:
        return getattr(clinic, name)(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments: {exc}"}
