"""One container for every tracking source: the provided ATD, Metrica's sample games, DFL.

Everything downstream (degradation, features, decoding) reads this layout only, so a model trained
on a public match sees exactly the arrays it will see on the provided one.

Coordinates are metres, x in [0, length] along the pitch, y in [0, width] across it, NaN where an
object is not observed. `label` uses the scorer's codes: 0 H, 1 A, 2 D, 3 X (unknown).
"""
from typing import NamedTuple

import numpy as np

H, A, D, X = 0, 1, 2, 3
FPS = 25


class Match(NamedTuple):
    name: str
    pitch: tuple              # (length, width) metres
    ball_xy: np.ndarray       # float32 [n, 2]
    home_xy: np.ndarray       # float32 [n, slots_h, 2]
    away_xy: np.ndarray       # float32 [n, slots_a, 2]
    label: np.ndarray         # int8 [n]
    period: np.ndarray        # int8 [n], 1 or 2, 0 outside play
    ball_z: np.ndarray        # float32 [n], NaN when the source has no height

    @property
    def n_frames(self):
        return len(self.label)

    def blocks(self):
        """[(period, start, stop)] half-open spans of each period, in frame order."""
        out = []
        for p in (1, 2):
            idx = np.flatnonzero(self.period == p)
            if len(idx):
                out.append((p, int(idx[0]), int(idx[-1]) + 1))
        return out


def save(path, m):
    np.savez_compressed(path, name=m.name, pitch=np.asarray(m.pitch, np.float32), ball_xy=m.ball_xy,
                        home_xy=m.home_xy, away_xy=m.away_xy, label=m.label, period=m.period,
                        ball_z=m.ball_z)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return Match(str(z["name"]), tuple(float(v) for v in z["pitch"]), z["ball_xy"], z["home_xy"],
                     z["away_xy"], z["label"], z["period"], z["ball_z"])
