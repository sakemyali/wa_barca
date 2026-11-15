"""DFL open dataset (Bassek et al., Scientific Data 2025, CC BY 4.0) -> Match.

Seven Bundesliga / 2. Bundesliga matches of TRACAB optical tracking at 25 Hz. The ball frames
carry BallStatus (1 in play, 0 dead) and BallPossession (1 home, 2 away) on every frame, which is
the same three-state labelling this challenge asks for, on other games.

https://doi.org/10.6084/m9.figshare.28196177
"""
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from ..schema import A, D, H, Match

# figshare file ids: match id -> (match information, positions)
FILES = {
    "J03WMX": (51643475, 51643514),
    "J03WN1": (51643472, 51643517),
    "J03WOH": (51643478, 51643520),
    "J03WOY": (51643481, 51643523),
    "J03WPY": (51643487, 51643526),
    "J03WQQ": (51643484, 51643529),
    "J03WR9": (51643490, 51643532),
}
URL = "https://ndownloader.figshare.com/files/{}"
SECTIONS = {"firstHalf": 1, "secondHalf": 2}

_FRAMESET = re.compile(rb'GameSection="(\w+)".*?TeamId="([^"]+)".*?PersonId="([^"]+)"')
_PLAYER = re.compile(rb'N="(\d+)".*? X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(rb'N="(\d+)".*? X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)".*?'
                   rb'BallPossession="(\d)" BallStatus="(\d)"')


def team_roles(info_stream):
    """{team_id: 'home' | 'guest'} from the match-information XML."""
    root = ET.parse(info_stream).getroot()
    return {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}


def parse_positions(stream, roles, name):
    """Stream the positions XML (one <Frame> per line) into a Match. Never holds the file in RAM."""
    pitch = (105.0, 68.0)
    tracks = {}      # (section, team, person) -> list of (n, x, y)
    ball = {}        # section -> list of (n, x, y, z, possession, status)
    current, is_ball = None, False
    for line in stream:
        if line.startswith(b"<Frame "):
            if current is None:
                continue
            if is_ball:
                m = _BALL.search(line)
                current.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), int(m[5]), int(m[6])))
            else:
                m = _PLAYER.search(line)
                current.append((int(m[1]), float(m[2]), float(m[3])))
        elif line.startswith(b"<FrameSet"):
            m = _FRAMESET.search(line)
            section, team, person = m[1].decode(), m[2].decode(), m[3].decode()
            is_ball = team.upper() == "BALL"
            if section not in SECTIONS or (not is_ball and team not in roles):
                current = None      # referees and anything outside the two halves
            elif is_ball:
                current = ball.setdefault(section, [])
            else:
                current = tracks.setdefault((section, team, person), [])
        elif line.startswith(b"<PitchSize"):
            m = re.search(rb'X="([\d.]+)" Y="([\d.]+)"', line)
            pitch = (float(m[1]), float(m[2]))

    persons = {role: sorted({k[2] for k in tracks if roles[k[1]] == role}) for role in ("home", "guest")}
    parts = []
    for section, period in SECTIONS.items():
        b = np.array(ball[section], dtype=np.float64)
        n0, n = int(b[0, 0]), len(b)
        assert np.array_equal(b[:, 0], np.arange(n0, n0 + n)), "ball frames are not contiguous"
        xy = {role: np.full((n, len(persons[role]), 2), np.nan, np.float32) for role in persons}
        for (sec, team, person), rows in tracks.items():
            if sec != section:
                continue
            r = np.array(rows, dtype=np.float64)
            idx = r[:, 0].astype(np.int64) - n0
            ok = (idx >= 0) & (idx < n)
            role = roles[team]
            xy[role][idx[ok], persons[role].index(person)] = r[ok, 1:3] + (pitch[0] / 2, pitch[1] / 2)
        label = np.where(b[:, 5] == 0, D, np.where(b[:, 4] == 1, H, A)).astype(np.int8)
        parts.append((b[:, 1:3] + (pitch[0] / 2, pitch[1] / 2), xy["home"], xy["guest"], label,
                      np.full(n, period, np.int8), b[:, 3]))
    cat = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    return Match(name, pitch, cat[0].astype(np.float32), cat[1], cat[2], cat[3], cat[4],
                 cat[5].astype(np.float32))


def fetch(match_id):
    """Download one match straight into a Match; the 400 MB XML is parsed as it streams."""
    info_id, pos_id = FILES[match_id]
    with urllib.request.urlopen(URL.format(info_id), timeout=120) as r:
        roles = team_roles(r)
    with urllib.request.urlopen(URL.format(pos_id), timeout=120) as r:
        return parse_positions(r, roles, "dfl_" + match_id)


if __name__ == "__main__":
    m = fetch(sys.argv[1])
    print(m.name, m.n_frames, m.home_xy.shape, m.away_xy.shape, np.bincount(m.label, minlength=3))
