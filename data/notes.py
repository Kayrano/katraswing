"""
Per-ticker trade notes — persisted to notes.json in the project root.
"""

import json
import os

_NOTES_FILE = os.path.join(os.path.dirname(__file__), "..", "notes.json")


def load_notes() -> dict:
    try:
        with open(_NOTES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_note(ticker: str) -> str:
    return load_notes().get(ticker.upper(), "")


def save_note(ticker: str, text: str) -> None:
    notes = load_notes()
    notes[ticker.upper()] = text.strip()
    with open(_NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def delete_note(ticker: str) -> None:
    notes = load_notes()
    notes.pop(ticker.upper(), None)
    with open(_NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)
