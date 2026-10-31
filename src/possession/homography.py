"""Per-frame 3x3 homography (normalised pitch 0-1 -> image pixels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import games
except ImportError:  # games.py sits at the repo root, which is not on sys.path under plain pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import games

MATRIX_COLUMNS = [f"h{r}{c}" for r in range(3) for c in range(3)]


def load_homography(source="tactical", data_dir=None):
    """(H [n,3,3] float64, NaN where invalid; reprojection_error [n]; valid bool [n])."""
    table = pd.read_csv(games.require(["homography"], data_dir=data_dir, source=source)[0])
    valid = table["valid"].to_numpy().astype(bool)
    H = table[MATRIX_COLUMNS].to_numpy(np.float64).reshape(-1, 3, 3)
    H[~valid] = np.nan  # the file already has NaN there; enforce it so no caller projects with a dead matrix
    return H, table["reprojection_error"].to_numpy(np.float64), valid


def _apply(M, a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    w = M[2, 0] * a + M[2, 1] * b + M[2, 2]
    return (M[0, 0] * a + M[0, 1] * b + M[0, 2]) / w, (M[1, 0] * a + M[1, 1] * b + M[1, 2]) / w


def project(H_frame, x, y):
    """Pitch (x, y) in 0-1 -> pixel (u, v). NaN matrix gives NaN."""
    return _apply(H_frame, x, y)


def unproject(H_frame, u, v):
    """Pixel (u, v) -> pitch (x, y) in 0-1."""
    return _apply(np.linalg.inv(H_frame), u, v)
