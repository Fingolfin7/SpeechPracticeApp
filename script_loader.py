import os
import json
import random

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
        idx = json.load(open(INDEX_FILE))
    else:
        idx = {"pos": 0, "order": []}

    # If script set changed, reshuffle
    if len(idx.get("order", [])) != len(files):
        idx["order"] = list(range(len(files)))
        random.shuffle(idx["order"])
        idx["pos"] = 0

    i = idx["order"][idx["pos"]]
    idx["pos"] = (idx["pos"] + 1) % len(files)
    json.dump(idx, open(INDEX_FILE, "w"), indent=2)

    path = os.path.join(SCRIPTS_DIR, files[i])
    text = open(path, encoding="utf-8").read().strip().lower()
    return files[i], text
