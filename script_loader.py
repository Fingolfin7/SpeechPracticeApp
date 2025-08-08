import os
import json
import random
from contextlib import suppress

SCRIPTS_DIR = "scripts"
INDEX_FILE  = "script_index.json"

def pick_next_script():
    """
    Round‐robin + shuffle picker from scripts/*.txt.
    Returns (filename, text).
    """
    files = sorted(
        f for f in os.listdir(SCRIPTS_DIR)
        if f.lower().endswith(".txt")
    )
    if os.path.exists(INDEX_FILE):
        with suppress(Exception):
            with open(INDEX_FILE, "r", encoding="utf-8") as fh:
                idx = json.load(fh)
    idx = idx if isinstance(idx := locals().get('idx', None), dict) else {"pos": 0, "order": []}

    # If script set changed, reshuffle
    if len(idx.get("order", [])) != len(files):
        idx["order"] = list(range(len(files)))
        random.shuffle(idx["order"])
        idx["pos"] = 0

    i = idx["order"][idx["pos"]]
    idx["pos"] = (idx["pos"] + 1) % len(files)
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2)

    path = os.path.join(SCRIPTS_DIR, files[i])
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip().lower()
    return files[i], text
