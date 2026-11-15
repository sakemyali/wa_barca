"""Find the opening kickoff of a half from player positions.

Why this exists. A scored half does not have to start at its kickoff: the provided second half
starts almost two minutes before it, with the players on the pitch and a ball being passed about.
To a model that looks very much like play. The public training matches cannot teach the
difference, because their recordings start at (or seconds before) the whistle. So it is settled by
the one thing that is true of every kickoff: the two teams stand on opposite sides of the halfway
line, and they stop doing so about a second and a half after the ball is struck.

The detector looks for the first moment at which a calm spell with (almost) everyone in their own
half gives way to sustained mixing, and steps back by the typical time the first players take to
cross the line. That delay is a constant measured on the public matches (experiments/calibration.md).
Everything before the kickoff is a dead ball.
"""
import warnings

import numpy as np

from .features import home_attacks_positive, roll_mean, rotate

TOLERANCE = 2.5          # metres past the halfway line before a player counts as "in the wrong half"
MIXED = 3                # this many players in the wrong half means play is on
CROSSING_DELAY = 32      # frames from the kick to the third player crossing (public-match median)
CALM_WINDOW = 125        # the 5 s before must be calm ...
MIXED_WINDOW = 250       # ... and the 10 s after must be mixed, each for 80% of the frames
SHARE = 0.8


def wrong_half_count(home_xy, away_xy, pitch):
    """Players standing beyond the halfway line, per frame. Either orientation of the pitch."""
    length = pitch[0]
    if not home_attacks_positive(home_xy, away_xy):
        home_xy, away_xy = rotate(home_xy, pitch), rotate(away_xy, pitch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ((home_xy[:, :, 0] > length / 2 + TOLERANCE).sum(1)
                + (away_xy[:, :, 0] < length / 2 - TOLERANCE).sum(1))


def opening_kickoff(home_xy, away_xy, pitch):
    """Index (within the half) of its opening kickoff; 0 when the half starts at the whistle."""
    wrong = wrong_half_count(home_xy, away_xy, pitch)
    mixed = (wrong >= MIXED).astype(np.float32)
    mixed_after = roll_mean(mixed, 0, MIXED_WINDOW - 1)
    # Share of calm frames in the 5 s before (fewer at the very start of the recording). A half
    # that opens already mixed has no calm before it at all: that is people walking to their
    # places, not a kickoff, and the search moves on to the first real calm-then-mixed moment.
    calm_before = np.nan_to_num(roll_mean(1.0 - mixed, -CALM_WINDOW, -1), nan=0.0)
    hit = np.flatnonzero((mixed > 0) & (mixed_after >= SHARE) & (calm_before >= SHARE))
    if not len(hit):
        return 0
    return max(int(hit[0]) - CROSSING_DELAY, 0)
