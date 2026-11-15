"""Metrica Sports' public sample games (three anonymised matches) -> Match.

https://github.com/metrica-sports/sample-data

These are other matches from the provider of the challenge data, and they are the closest public
thing to its ground truth: corrected 25 Hz tracking with synchronised events. Labels are derived
mechanically, with no hand annotation:

  dead   (a) the corrected ball track is absent (NaN) for more than a blink -- Metrica blanks the
         ball while it is out of play -- or (b) the stretch between the end of the last event and
         the restart that follows it (any SET PIECE, or a goal kick). Rule (b) catches the
         stoppages where the ball was left visible, lying still, waiting for a free kick or a
         goal kick.
  owner  the team of the most recent on-ball event (pass, carry, shot, recovery, ball lost, set
         piece, ball out); after a stoppage the first event decides, so the restart is owned from
         its first live frame
"""
import json
import os
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..schema import A, D, H, Match, save

RAW_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawEventsData.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawEventsData.csv"],
    3: ["Sample_Game_3/Sample_Game_3_tracking.txt", "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_events.json"],
}
PITCH = (105.0, 68.0)
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE", "BALL OUT"}
# A corrected ball track that vanishes for a few frames inside live play is an interpolation
# blink, not a stoppage: real stoppages last seconds.
MIN_DEAD_FRAMES = 10


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def derive_labels(ball_nan, period, ev):
    """Per-frame H/A/D. `ev` is a dict of equal-length arrays sorted by start frame:
    start, end (row indices), team (0/1), on_ball (bool), restart (bool), card (bool)."""
    n = len(ball_nan)
    dead = np.zeros(n, bool)
    for s, e in zip(*_runs(ball_nan)):
        if e - s >= MIN_DEAD_FRAMES:
            dead[s:e] = True
    first_of_period = {p: int(np.flatnonzero(period == p)[0]) for p in (1, 2) if (period == p).any()}
    last_end = None            # end frame of the latest non-card event seen so far
    for i in range(len(ev["start"])):
        s0 = int(ev["start"][i])
        if ev["restart"][i]:
            p0 = first_of_period.get(int(period[min(max(s0, 0), n - 1)]), 0)
            begin = p0 if last_end is None or last_end < p0 else last_end
            if s0 > begin:
                dead[begin:s0] = True
        if not ev["card"][i]:
            e0 = max(s0, int(ev["end"][i]))
            last_end = e0
    # ownership: team of the last on-ball event
    ob = ev["on_ball"]
    ev_frame, ev_team = ev["start"][ob], ev["team"][ob]
    owner = np.full(n, -1, np.int8)
    idx = np.searchsorted(ev_frame, np.arange(n), side="right") - 1
    has = idx >= 0
    owner[has] = ev_team[idx[has]]
    # restart: from the first live frame after a stoppage (or a period start) the next event owns it
    nxt = np.searchsorted(ev_frame, np.arange(n), side="left")
    starts = set(np.flatnonzero(np.diff(np.concatenate(([1], dead.astype(np.int8)))) == -1).tolist())
    starts |= set(first_of_period.values())
    for s in sorted(starts):
        j = nxt[s]
        if j < len(ev_frame):
            stop = int(ev_frame[j])
            if stop > s and not dead[s:stop].any():
                owner[s:stop] = ev_team[j]
    label = np.where(dead | (owner < 0), D, owner).astype(np.int8)
    return label


def _fill_blinks(ball, dead):
    """Linear fill of the short NaN gaps that were judged live, so the track is continuous."""
    ball = ball.copy()
    nan = np.isnan(ball[:, 0]) & ~dead
    if nan.any():
        t = np.arange(len(ball))
        ok = ~np.isnan(ball[:, 0])
        for c in (0, 1):
            ball[nan, c] = np.interp(t[nan], t[ok], ball[ok, c])
    return ball


def load_csv_game(folder, game):
    home_csv, away_csv, events_csv = [os.path.join(folder, os.path.basename(f)) for f in FILES[game]]
    home = pd.read_csv(home_csv, skiprows=2)
    away = pd.read_csv(away_csv, skiprows=2)
    assert len(home) == len(away)
    period = home["Period"].to_numpy(np.int8)
    hx = home.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(home), -1, 2)
    ax = away.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(away), -1, 2)
    ball = home.iloc[:, -2:].to_numpy(np.float32)
    ev = pd.read_csv(events_csv).sort_values(["Start Frame", "End Frame"], kind="stable")
    sub = ev["Subtype"].fillna("").astype(str)
    events = {
        "start": ev["Start Frame"].to_numpy(np.int64) - 1,          # frames count from 1
        "end": ev["End Frame"].to_numpy(np.int64) - 1,
        "team": np.where(ev["Team"].to_numpy() == "Home", H, A),
        "on_ball": ev["Type"].isin(ON_BALL).to_numpy(),
        "restart": ((ev["Type"] == "SET PIECE") | sub.str.contains("GOAL KICK")).to_numpy(),
        "card": (ev["Type"] == "CARD").to_numpy(),
    }
    return _finish("metrica_sample_{}".format(game), ball, hx, ax, period, events)


def load_epts_game(folder, game=3):
    txt, meta, events = [os.path.join(folder, os.path.basename(f)) for f in FILES[game]]
    root = ET.parse(meta).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    players = [p.get("id") for p in root.iter("Player")]
    team_of = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    # The column layout changes at every substitution: each DataFormatSpecification lists the
    # players on the pitch, in file order, for one span of frames.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [channel_player[r] for r in refs[::2]]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    ids, rows = [], []
    with open(txt) as f:
        for line in f:
            fid, body, ball = line.strip().split(":")
            ids.append(int(fid))
            rows.append(np.array((body.replace(";", ",") + "," + ball).split(","), dtype=np.float32))
    ids = np.array(ids)
    n = len(ids)
    slot = {pid: i for i, pid in enumerate(players)}
    pl = np.full((n, len(players), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    for start, stop, order in specs:
        cols = [slot[pid] for pid in order]
        for i in np.flatnonzero((ids >= start) & (ids <= stop)):
            row = rows[i]
            assert len(row) == 2 * len(order) + 2, "frame {}: {} values".format(ids[i], len(row))
            pl[i, cols] = row[:-2].reshape(-1, 2)
            ball[i] = row[-2:]
    teams_of_slot = np.array([team_of[pid] for pid in players])
    period = np.zeros(n, np.int8)
    period[(ids >= int(params["first_half_start"])) & (ids <= int(params["first_half_end"]))] = 1
    period[(ids >= int(params["second_half_start"])) & (ids <= int(params["second_half_end"]))] = 2
    data = json.load(open(events))["data"]
    row_of = {fid: i for i, fid in enumerate(ids)}
    rows_ = []
    for e in data:
        if e["start"]["frame"] not in row_of:
            continue
        subs = e.get("subtypes") or []
        subs = [subs] if isinstance(subs, dict) else subs
        names = {x["name"] for x in subs}
        kind = e["type"]["name"]
        end = row_of.get(e["end"]["frame"], row_of[e["start"]["frame"]])
        rows_.append((row_of[e["start"]["frame"]], end, team_ids.index(e["team"]["id"]), kind in ON_BALL,
                      kind == "SET PIECE" or "GOAL KICK" in names, kind == "CARD"))
    rows_.sort(key=lambda r: (r[0], r[1]))
    cols = list(zip(*rows_))
    events_ = {"start": np.array(cols[0]), "end": np.array(cols[1]), "team": np.array(cols[2]),
               "on_ball": np.array(cols[3]), "restart": np.array(cols[4]), "card": np.array(cols[5])}
    return _finish("metrica_sample_{}".format(game), ball, pl[:, teams_of_slot == 0],
                   pl[:, teams_of_slot == 1], period, events_)


def _finish(name, ball, hx, ax, period, events):
    label = derive_labels(np.isnan(ball[:, 0]), period, events)
    ball = _fill_blinks(ball, label == D)
    scale = np.array(PITCH, np.float32)
    return Match(name, PITCH, ball * scale, hx * scale, ax * scale, label, period,
                 np.full(len(label), np.nan, np.float32))


def download(folder):
    os.makedirs(folder, exist_ok=True)
    for files in FILES.values():
        for rel in files:
            path = os.path.join(folder, os.path.basename(rel))
            if not os.path.isfile(path):
                urllib.request.urlretrieve(RAW_URL + rel, path)


def prepare(out_dir, raw_dir=None):
    raw_dir = raw_dir or os.path.join(out_dir, "raw_metrica")
    download(raw_dir)
    for game in FILES:
        m = load_epts_game(raw_dir, game) if game == 3 else load_csv_game(raw_dir, game)
        path = os.path.join(out_dir, m.name + ".npz")
        save(path, m)
        print("{}: {} frames, H/A/D {} -> {}".format(m.name, m.n_frames,
                                                     np.bincount(m.label, minlength=3).tolist(), path))
