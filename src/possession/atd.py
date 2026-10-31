"""Fast loader for the Metrica FIFA/EPTS Automatic Tracking Data, cached to .npz.

Raw line: '<frame_id>:<slot>;<slot>;...:<ball>', each slot and the ball 'x,y,speed' in 0-1, NaN if
missing. frame_id counts from 1 and video_frame = frame_id - 1. A slot is a tracklet, not a person.
"""
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import games  # noqa: E402  (games.py sits at the repo root)


class ATD(NamedTuple):
    frame: np.ndarray          # int64 [n], video frames 0..n-1
    ball_xy: np.ndarray        # float32 [n,2], metres, NaN where missing
    players_xy: np.ndarray     # float32 [n,slots,2], metres
    players_speed: np.ndarray  # float32 [n,slots]
    team_of_slot: np.ndarray   # int8 [slots], 0 home, 1 away
    home_name: str
    away_name: str
    pitch: tuple               # (length, width) metres
    n_frames: int


def parse_atd(raw, n_slots):
    """Parse raw tracking text into (frame_ids, ball_xy, players_xy, players_speed), pitch units 0-1."""
    # shortcut: one flat float parse of the whole text, needs ~8x file size in RAM; chunk if that bites.
    flat = np.fromstring(raw.strip().replace(":", ",").replace(";", ",").replace("\n", ","), sep=",")
    cols = 1 + 3 * (n_slots + 1)
    if flat.size % cols:
        raise ValueError("tracking text is not {} values per frame ({} values)".format(cols, flat.size))
    a = flat.reshape(-1, cols)
    ids = a[:, 0].astype(np.int64)
    players = a[:, 1:-3].reshape(len(a), n_slots, 3)
    return ids, a[:, -3:-1], players[:, :, :2], players[:, :, 2]


def parse_meta(xml_path):
    """(team_of_slot, home_name, away_name) from the metadata XML; first Team is home."""
    root = ET.parse(xml_path).getroot()
    teams = root.findall(".//Teams/Team")
    team_idx = {t.get("id"): i for i, t in enumerate(teams[:2])}
    team_of_slot = np.array([team_idx[p.get("teamId")] for p in root.findall(".//Players/Player")],
                            dtype=np.int8)
    return team_of_slot, teams[0].findtext("Name"), teams[1].findtext("Name")


def _to_atd(ids, ball, pxy, spd, team_of_slot, home, away, pitch):
    scale = np.array(pitch, dtype=np.float32)
    return ATD(frame=ids - 1, ball_xy=(ball * scale).astype(np.float32),
               players_xy=(pxy * scale).astype(np.float32), players_speed=spd.astype(np.float32),
               team_of_slot=team_of_slot, home_name=home, away_name=away, pitch=tuple(pitch),
               n_frames=len(ids))


def load_atd(source="tactical", data_dir=None, use_cache=True):
    xml_path, txt_path = games.require(["atd_meta", "atd_raw"], data_dir=data_dir, source=source)
    pitch = games.GAMES[games.GAME]["pitch"]
    team_of_slot, home, away = parse_meta(xml_path)
    cache_dir = os.path.join(data_dir or games.DATA_DIR, ".cache")
    cache = os.path.join(cache_dir, games.SOURCES[source]["stem"] + "_atd.npz")

    src_stat = os.stat(txt_path)
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) >= src_stat.st_mtime:
        with np.load(cache) as z:
            if "src_size" in z and int(z["src_size"]) == src_stat.st_size:  # older cache -> rebuild
                return ATD(z["frame"], z["ball_xy"], z["players_xy"], z["players_speed"],
                           team_of_slot, home, away, tuple(pitch), len(z["frame"]))

    with open(txt_path, "r") as f:
        parts = _to_atd(*parse_atd(f.read(), len(team_of_slot)), team_of_slot, home, away, pitch)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, frame=parts.frame, ball_xy=parts.ball_xy, players_xy=parts.players_xy,
                 players_speed=parts.players_speed, src_size=src_stat.st_size)
    return parts


def check_alignment(data, source="tactical"):
    """Assert frames are exactly 0..n-1 (raw ids 1..n, none missing) and n matches the source."""
    assert np.array_equal(data.frame, np.arange(data.n_frames)), "frame ids are not 1..n contiguous"
    expected = games.source_frames(source)
    assert data.n_frames == expected, "n_frames {} != {}".format(data.n_frames, expected)
    cov = 100.0 * np.isfinite(data.ball_xy).all(axis=1).mean()
    print("{}: {} frames, ball coverage {:.2f}%".format(source, data.n_frames, cov))
    return cov


if __name__ == "__main__":
    t0 = time.time()
    d = load_atd("tactical")
    print("load {:.1f}s".format(time.time() - t0))
    check_alignment(d)
