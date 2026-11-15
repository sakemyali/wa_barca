"""Load matches, from any source, as the common Match layout."""
import glob
import os

import numpy as np

import games

from . import schema
from .atd import load_atd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(ROOT, "data_external")


def provided(data_dir=None):
    """The provided match (tactical ATD) as a Match. Labels are all X: there are none to read."""
    a = load_atd("tactical", data_dir=data_dir)
    n = a.n_frames
    period = np.zeros(n, np.int8)
    for half, first, last in games.half_video_blocks(games.GAME):
        period[first:last + 1] = half
    return schema.Match(games.GAME, a.pitch, a.ball_xy, a.players_xy[:, a.team_of_slot == 0],
                        a.players_xy[:, a.team_of_slot == 1], np.full(n, schema.X, np.int8), period,
                        np.full(n, np.nan, np.float32))


def external(prefix=""):
    """Every converted public match whose name starts with `prefix` ('dfl', 'metrica', or '')."""
    paths = sorted(glob.glob(os.path.join(EXTERNAL_DIR, prefix + "*.npz")))
    if not paths:
        raise IOError("no public matches in {}; run python scripts/prepare_external.py".format(EXTERNAL_DIR))
    return [schema.load(p) for p in paths]
