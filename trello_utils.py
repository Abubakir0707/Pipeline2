"""
trello_utils.py — Trello REST API helpers for Lead Scoring app.

All functions accept api_key and token explicitly so they work
both locally (from st.secrets) and on Streamlit Cloud.
"""

import requests
from typing import Optional

BASE = "https://api.trello.com/1"


# ─────────────────────────────────────────────
# Board / List helpers
# ─────────────────────────────────────────────

def get_boards(api_key: str, token: str) -> list[dict]:
    """Return all boards for the authenticated member."""
    r = requests.get(
        f"{BASE}/members/me/boards",
        params={"key": api_key, "token": token, "fields": "id,name"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_lists(board_id: str, api_key: str, token: str) -> list[dict]:
    """Return all open lists on a board."""
    r = requests.get(
        f"{BASE}/boards/{board_id}/lists",
        params={"key": api_key, "token": token, "filter": "open", "fields": "id,name"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_cards(list_id: str, api_key: str, token: str) -> list[dict]:
    """Return all cards in a list with name + description."""
    r = requests.get(
        f"{BASE}/lists/{list_id}/cards",
        params={"key": api_key, "token": token, "fields": "id,name,desc,dateLastActivity,url"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# Card creation / update
# ─────────────────────────────────────────────

def create_card(
    list_id: str,
    name: str,
    desc: str,
    api_key: str,
    token: str,
    label_color: Optional[str] = None,   # e.g. "red", "green", "blue"
) -> dict:
    """Create a card and return the created card object."""
    payload = {
        "key": api_key,
        "token": token,
        "idList": list_id,
        "name": name,
        "desc": desc,
    }
    r = requests.post(f"{BASE}/cards", params=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def create_board_and_list(
    board_name: str,
    list_name: str,
    api_key: str,
    token: str,
) -> tuple[str, str]:
    """
    Create a new Trello board with a single list.
    Returns (board_id, list_id).
    """
    # Create board
    rb = requests.post(
        f"{BASE}/boards/",
        params={"key": api_key, "token": token, "name": board_name,
                "defaultLists": "false"},
        timeout=10,
    )
    rb.raise_for_status()
    board_id = rb.json()["id"]

    # Create list on that board
    rl = requests.post(
        f"{BASE}/lists",
        params={"key": api_key, "token": token,
                "name": list_name, "idBoard": board_id},
        timeout=10,
    )
    rl.raise_for_status()
    list_id = rl.json()["id"]

    return board_id, list_id
