# Experiment log

Every number here is measured on **held-out public matches** (other games), degraded to resemble
the provided tracking, with the organisers' scorer. Three folds over ten matches; each fold holds
out one Metrica sample game and two or three DFL matches (`scripts/run_cv.py`). Decoder settings
are chosen on the pooled out-of-fold probabilities (`scripts/tune_decoder.py`). The provided
match's public five minutes are never used to choose anything; where they are quoted it says so.

`results.jsonl` has one line per run (options, runtime, date); `decoder_<tag>.json` the tuned
decoder of each run; `logs/` the raw output of the final rebuild.

## Summary

| tag | what changed | spellF1 | macroF1 | note |
|---|---|---|---|---|
| — | organisers' baseline (nearest player, majority vote ±50) on the same matches | 0.248 | 0.652 | 0.243 / 0.672 published for the private ground truth |
| v1 | first complete pipeline; ball smoothing σ = 4 | 0.667 | 0.916 | easier simulation, **not comparable** with the rows below |
| v2a | degradation re-measured: σ = 7, positional offset | 0.637 | 0.908 | same features as v1; the drop is the harder data |
| v2b | + de-smoothed ball position, touch chain | 0.651 | 0.912 | +0.015 |
| **v3** | + three event detectors, event-aware decoder | **0.659** | **0.915** | +0.008; **submitted configuration** |
| v4a | v3 with equal weight per possession (exponent 1) | 0.657 | 0.910 | no gain |
| v4b | v3 with unweighted training | 0.646 | 0.914 | −0.014: the weighting earns its place |
| tcn_stack | temporal convolutional network on features + tree probabilities; one fold (four held-out matches) | 0.667 | 0.926 | v3 on the same four matches: 0.655 / 0.918. +0.012, on all four matches; not adopted (see below) |

Per held-out match (v3): dfl_J03WMX 0.621 · J03WN1 0.642 · J03WOH 0.678 · J03WOY 0.641 ·
J03WPY 0.706 · J03WQQ 0.665 · J03WR9 0.683 · metrica_sample_1 0.668 · 2 0.631 · 3 0.657.

## The temporal network (tcn_stack)

`scripts/run_tcn.py tcn_stack inputs=stack oof=v3 epochs=6 folds=1` — 7 dilated residual blocks,
64 channels, 260 inputs per frame (255 features + 5 tree probabilities as logits), trained on the
out-of-fold probabilities of six matches (both team orders), 426 steps of 12 × 2,000 frames,
12 minutes on 8 CPU cores. Tested on dfl_J03WMX, dfl_J03WOY, dfl_J03WR9 and metrica_sample_1,
each decoder tuned on those four matches:

| | spellF1 | macroF1 | per match |
|---|---|---|---|
| tree models (v3) | 0.655 | 0.918 | 0.628 · 0.639 · 0.685 · 0.666 |
| network on top (hybrid) | 0.667 | 0.926 | 0.641 · 0.646 · 0.705 · 0.677 |

Frame level, network: dead-ball accuracy 0.973–0.985, team accuracy on live frames 0.891–0.926.
A consistent gain, but a small one (the metric's noise band is 0.04), from a component that does
not regenerate bit-identically across machines. Not in the submission. The three-fold run and
the features-only and probabilities-only variants were not run for lack of time.

## Decoders on the same probabilities (v3)

`scripts/ablate_decoder.py v3`

| decoder | spellF1 | macroF1 | possessions predicted / true |
|---|---|---|---|
| per-frame argmax | 0.422 | 0.910 | 1.98 |
| argmax + majority vote ±12 frames | 0.617 | 0.911 | 1.00 |
| argmax + majority vote ±25 | 0.638 | 0.911 | 0.86 |
| argmax + majority vote ±50 | 0.602 | 0.908 | 0.70 |
| argmax + majority vote ±100 | 0.487 | 0.890 | 0.52 |
| Viterbi, tuned (temper 0.3, costs 2 / 6 / 14) | 0.651 | 0.912 | 0.82 |
| Viterbi + event-aware costs, tuned (0.3, 4.5 / 14 / 6, gains 1 / 1 / 1) | 0.659 | 0.915 | 0.79 |

The optimum is flat: the twelve best of 1,080 settings are within 0.002, and the two stoppage
costs can be exchanged freely. The tuned setting is committed as `models/decoder.json`.

## Where the points are lost (v3)

`scripts/error_analysis.py v3`

| | spellF1 |
|---|---|
| as predicted | 0.659 |
| perfect alive/dead call, our team call | 0.711 |
| perfect team call, our alive/dead call | 0.938 |

| IoU threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| F1 | 0.761 | 0.723 | 0.670 | 0.608 | 0.534 |

| true possession length | n | recall at IoU 0.5 | predictions of that length | their precision |
|---|---|---|---|---|
| under 1 s | 533 | 0.073 | 78 | 0.436 |
| 1–2 s | 712 | 0.341 | 397 | 0.562 |
| 2–4 s | 926 | 0.545 | 755 | 0.668 |
| 4–10 s | 1,066 | 0.768 | 1,039 | 0.782 |
| over 10 s | 980 | 0.931 | 1,041 | 0.907 |

What happens to the missed ones: under 1 s — 55% given to the other team, 33% right team but
merged into a much longer predicted possession, 11% called dead ball; 1–2 s — 51% / 35% / 9%;
2–4 s — 44% / 43% / 9%; 4–10 s — 28% / 49% / 8%. Mean probability of the correct team inside a
missed possession: 0.45, 0.52, 0.59, 0.70 — the model is undecided there, not confidently wrong.

Boundary timing of matched possessions, team changes: median error 11 frames before the event
detectors (v2a), 4 frames with them (v3; 53% within 5 frames, 75th percentile 32). Stoppage and
restart boundaries: median 4 frames, 75th percentile 23.

## Single components (first experiments)

Run before the degradation existed, to see what transfers between the two public sources: players
as published, ball smoothed with σ = 4, no loss of visibility (`experiments/early/`, kept as run;
frame accuracy).

| model | train → test | accuracy |
|---|---|---|
| alive/dead, players only | DFL → Metrica | 0.884–0.936 |
| | Metrica → DFL | 0.921–0.934 |
| | mixed → held out | 0.887–0.954 |
| team, ball + players (live frames) | DFL → Metrica | 0.922–0.964 |
| | Metrica → DFL | 0.897–0.925 |
| | mixed → held out | 0.908–0.932 |
| team, players only (no ball at all) | mixed → held out | 0.838–0.878 |
| nearest player to the ball (no model) | — | 0.746–0.803 |

Team accuracy by distance to the nearest change of label (mixed): 0.74–0.80 within 12 frames,
0.87–0.93 at 12–50, 0.97–0.98 at 50–150, 0.999 beyond on the DFL matches. On Metrica game 1 the
last figure was 0.927, which is how its mislabelled stoppages (a ball left visible, lying still
for 161 seconds) were found and the label rule corrected.

## Checks that are not experiments

- **Simulation realism** (`scripts/baseline_check.py`): the organisers' baseline on the degraded
  public matches against its published full-match scores — raw 0.089 vs 0.049 spellF1, ±25 0.227
  vs 0.200, ±50 0.248 vs 0.243, ±125 0.210 vs 0.224; macroF1 0.59–0.66 vs 0.64–0.68.
- **Calibration** (`scripts/calibrate.py`, output in `calibration.md`).
- **Domain check** (`scripts/domain_check.py`): a classifier separates provided from degraded
  public frames perfectly (AUC 1.0), as it would any single match from others. The features it
  uses are tactical fingerprints (goalkeeper depth, distance between the teams' deepest players)
  and the strength of ball redirections; their ranges overlap, none is out of the training range.
- **Kickoff detector** on the public matches: fires in 5 of 20 halves (recordings that start
  before the whistle), 0 / 13 / 21 / 31 frames late and 426 on a retaken kickoff; validation score
  unchanged. Provided match: 1 and 13 frames from the organisers' published anchors.
- **Smart Tagging against our team call** (no labels): build up 0.93, progression 0.89, ball in
  final third 0.76, creating chances 0.75, attacking transition 0.73, ball in the box 0.70, long
  ball 0.69, players in the box 0.62 — share of live frames inside the clip that we give to the
  tagged team. Not used (see the technical document).
- **Public five minutes of the provided match** (pipeline check, 17 possessions): v1 0.611 /
  0.919, v3 0.736 / 0.929 (the final rebuild is the same file). Baselines there: 0.038–0.424.
  Scorer's bootstrap interval for the submitted file: 0.53–0.89.

## Reproducibility check

The chain `bootstrap.py` → `train.py` → `predict.py` was run from scratch with the final code. It
produced, byte for byte, the same CSV as the build made earlier the same day from separately
trained models (sha256 `3bd35b750b8c3977…`). `logs/rebuild_1.sha256` holds the hashes of that CSV
and of the model files; `logs/final_*.log` the output of the run. A further repeat was started
and interrupted before it finished.
