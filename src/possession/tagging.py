"""Smart Tagging (SportsCode XML) as a table of tactical-video frame intervals."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import games
except ImportError:  # games.py sits at the repo root, which is not on sys.path under plain pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import games

TEAMS = {"Barcelona": "H", "Atletic Club": "A"}  # anything else (N/A) -> None
GROUPS = {"Type": "type", "Side": "side", "Max Players in the box": "max_players_in_box",
          "Direction of ball entry": "direction_of_ball_entry"}


def parse_tagging(path, fps=games.GAMES[games.GAME]["fps"]):
    rows = []
    for inst in ET.parse(path).getroot().iter("instance"):
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.iter("label")}
        half = labels.get("Half")
        row = {
            "id": int(inst.findtext("ID")),
            "code": inst.findtext("code").strip().lower(),
            "team": TEAMS.get(labels.get("Team")),
            "half": int(half[0]) if half else None,  # "1st Half" -> 1
            "start_frame": round(float(inst.findtext("start")) * fps),
            "end_frame": round(float(inst.findtext("end")) * fps),
        }
        row.update({col: labels.get(group) for group, col in GROUPS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_tagging(data_dir=None):
    return parse_tagging(games.require(["smart_tagging"], data_dir=data_dir)[0])


def frame_mask(df, n_frames):
    """Boolean [n_frames], True where any row's [start_frame, end_frame] covers the frame."""
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in zip(df["start_frame"], df["end_frame"]):
        mask[max(s, 0):max(e + 1, 0)] = True
    return mask
