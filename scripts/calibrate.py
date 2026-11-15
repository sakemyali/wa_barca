"""Every label-free measurement the degradation is calibrated on, in one place.

    python scripts/calibrate.py

Prints, for the provided match and for degraded public matches side by side, the statistics that
`degrade.py` is tuned to reproduce (kick pulse width, velocity change, distance to the nearest
player, how often the nearest player changes team), plus the facts about the provided tracking
that the design rests on (how the ball track comes and goes, what the players do meanwhile, and
that airborne balls are not displaced). Nothing here reads a label of the provided match: "live"
frames are the ones the players-only bootstrap calls live.
"""
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import degrade, features, matches, pipeline  # noqa: E402
from possession.schema import D  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = os.path.join(games.DATA_DIR, ".cache")


def pulse_width(ball, live):
    """Full width at half maximum, in frames, of the mean acceleration pulse around strong kicks."""
    v = np.diff(ball, axis=0) * 25
    acc = np.linalg.norm(np.diff(v, axis=0) * 25, axis=1)
    acc = np.where(np.isfinite(acc), acc, 0.0)
    peaks = np.flatnonzero((acc == maximum_filter1d(acc, 31)) & (acc > 15) & live[1:-1])
    peaks = peaks[(peaks > 20) & (peaks < len(acc) - 20)]
    profile = np.stack([acc[p - 15:p + 16] / acc[p] for p in peaks]).mean(0)
    return int((profile > 0.5).sum()), len(peaks)


def ball_stats(m, live):
    """Calibration statistics of one match over its live, ball-visible frames."""
    widths, dv, dmin, speed, switches, pairs = [], [], [], [], 0, 0
    for _, s, e in m.blocks():
        ball = m.ball_xy[s:e].astype(np.float64)
        f = features.ball_features(m.ball_xy[s:e], m.home_xy[s:e], m.away_xy[s:e], m.pitch)
        ok = live[s:e] & np.isfinite(ball[:, 0])
        widths.append(pulse_width(np.where(np.isfinite(ball), ball, 0.0), ok)[0])
        dv.append(f["b_dv"][ok])
        dmin.append(f["d_min"][ok])
        speed.append(f["b_sp"][ok])
        near = np.where(np.isfinite(f["d_diff"]), (f["d_diff"] < 0).astype(np.int8), -1)
        both = ok[1:] & ok[:-1] & (near[1:] >= 0) & (near[:-1] >= 0)
        switches += int(((near[1:] != near[:-1]) & both).sum())
        pairs += int(both.sum())
    dv, dmin, speed = np.concatenate(dv), np.concatenate(dmin), np.concatenate(speed)
    return {"pulse width (frames)": "{:.0f}".format(np.mean(widths)),
            "velocity change over 0.5 s, pct 50/75/90/99": np.nanpercentile(dv, [50, 75, 90, 99]).round(1).tolist(),
            "share with a player within 1 m": "{:.3f}".format(np.nanmean(dmin < 1.0)),
            "ball speed, pct 50/75/90/99": np.nanpercentile(speed, [50, 75, 90, 99]).round(1).tolist(),
            "nearest-team switches per 1000 frames": "{:.1f}".format(1000.0 * switches / pairs)}


def provided_facts(m, dead):
    inside = m.period > 0
    visible = np.isfinite(m.ball_xy[:, 0])
    print("PROVIDED MATCH")
    print("  graded frames {}; ball track present on {:.1%}".format(inside.sum(), visible[inside].mean()))
    gaps, shown = [], []
    for _, s, e in m.blocks():
        gaps += [b - a for a, b in degrade.runs(~visible[s:e])]
        shown += [b - a for a, b in degrade.runs(visible[s:e])]
    gaps = np.array(gaps)
    print("  {} visible runs, {} gaps; median gap {:.1f} s; {:.1%} of missing frames are in gaps of 1 s or more".format(
        len(shown), len(gaps), np.median(gaps) / 25, gaps[gaps >= 25].sum() / gaps.sum()))
    allxy = np.concatenate([m.home_xy, m.away_xy], 1)
    v = features.central_velocity(allxy, 12)
    med = np.nanmedian(np.hypot(v[:, :, 0], v[:, :, 1]), 1)
    print("  median player speed: {:.2f} m/s while the ball track is missing, {:.2f} m/s while present; "
          "{:.0%} of missing frames are under 1.5 m/s".format(
              np.nanmedian(med[inside & ~visible]), np.nanmedian(med[inside & visible]),
              np.nanmean(med[inside & ~visible] < 1.5)))
    n_vis = np.isfinite(allxy[:, :, 0]).sum(1)[inside]
    print("  visible players per graded frame: mean {:.1f}, median {:.0f}, 5th percentile {:.0f}".format(
        n_vis.mean(), np.median(n_vis), np.percentile(n_vis, 5)))
    # airborne balls: an uncorrected ground projection would bulge away from the camera
    ball = m.ball_xy.astype(np.float64)
    k = 4
    vel = np.full_like(ball, np.nan)
    acc = np.full_like(ball, np.nan)
    vel[k:-k] = (ball[2 * k:] - ball[:-2 * k]) * 25 / (2 * k)
    acc[k:-k] = (ball[2 * k:] - 2 * ball[k:-k] + ball[:-2 * k]) * (25.0 / k) ** 2
    fast = inside & (np.linalg.norm(vel, axis=1) > 14) & np.isfinite(acc[:, 1])
    print("  lateral acceleration of balls faster than 14 m/s: mean {:+.2f} m/s^2 over {} frames "
          "(an uncorrected projection from this camera would give about +7)".format(acc[fast, 1].mean(), fast.sum()))
    bank = degrade.visibility_bank(visible[m.period > 0], dead, pipeline.blocks_of(m.period[m.period > 0]))
    for key in ("dead", "alive"):
        masks = bank[key]
        share = np.array([x.mean() for x in masks])
        print("  bank[{}]: {} segments; ball visible on {:.1%} of their frames; {:.0%} never visible, {:.0%} always".format(
            key, len(masks), sum(x.sum() for x in masks) / sum(len(x) for x in masks), (share == 0).mean(), (share == 1).mean()))
    return bank


def main():
    prov = matches.provided()
    z = np.load(os.path.join(CACHE, "bootstrap.npz"))
    dead_rows = z["dead"]
    bank = provided_facts(prov, dead_rows)
    live = np.zeros(prov.n_frames, bool)
    live[z["rows"]] = ~dead_rows
    table = {"provided match": ball_stats(prov, live)}
    ext = matches.external()
    for i in (1, 5, 7, 9):
        d = degrade.degrade(ext[i], bank, seed=2 * i)
        table[ext[i].name + " (degraded)"] = ball_stats(d, ext[i].label < D)
    table[ext[5].name + " (as published)"] = ball_stats(ext[5], ext[5].label < D)
    print("\nCALIBRATION, live frames with a visible ball")
    for stat in next(iter(table.values())):
        print("  " + stat)
        for name, row in table.items():
            print("      {:<32} {}".format(name, row[stat]))


if __name__ == "__main__":
    main()
