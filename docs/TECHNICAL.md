# Per-frame ball possession from automatic tracking

Technical document — More than a Hack 2027, phase 1 — repository `wa_barca`

**Result in one paragraph.** For every frame of FC Barcelona vs Athletic Club we output `H`, `A`
or `D` (and `X` only outside the two graded halves). The system is trained entirely on public
tracking of **other** matches, degraded to resemble the provided automatic tracking, and scores
**spellF1 0.659 / macroF1 0.915** on held-out public matches that it never saw (3-fold, ten
matches, organisers' scorer). On those same degraded matches the organisers' reference baseline
scores 0.248 / 0.652, against the 0.243 / 0.672 the organisers report for it on the private ground
truth — which is our evidence that the simulated matches are about as hard as the real one. No
label of the provided match was used for training, tuning or model selection.

---

## 1. The problem as we read it

Three states per frame: home in possession, away in possession, dead ball. The ranking metric,
`spellF1`, matches predicted possessions to true ones by temporal IoU, so what is scored is the
**segmentation** — where each possession starts and stops — much more than the per-frame label.
Two things follow, and they shaped everything:

- A missed two-second possession costs more than itself: the possessions either side of it are
  merged into one prediction, which can match at most one of them. An invented one does the
  mirror-image damage. So the system must decide on short possessions; it cannot smooth them away.
- The public five minutes show that the ground truth is event-driven: possession passes at the
  *first touch* of the team that takes over (in one case for 0.76 s), not when control has become
  obvious. Labelling is therefore retrospective, and features and decoder look both ways in time.
  The task is offline, so that is legitimate.

## 2. What the provided tracking is actually like

All measured on the provided match without labels (`experiments/calibration.md` has the commands).

| fact | value | consequence |
|---|---|---|
| ball track present | 59.3% of graded frames, in **149 long runs** separated by 150 gaps (median gap ≈ 11 s; 99.9% of missing frames sit in gaps longer than 1 s) | the track does not flicker; it is *switched off* for long stretches |
| what players do while it is missing | median player speed 0.97 m/s, against 1.62 m/s while it is present; 87% of missing frames have a median speed under 1.5 m/s | "missing" mostly means stoppage — but not always, in either direction |
| ball smoothing | a kick is an acceleration pulse ≈ 19 frames wide (0.8 s); in raw optical tracking it is 1 frame | sharp touches are blurred; the filtered ball cuts the corner at every kick |
| airborne balls | no parallax signature: mean lateral acceleration −0.4 m/s² above 14 m/s, where an uncorrected ground projection from the recovered camera pose (45 m behind the touchline, 30 m up) would give ≈ +7 | ball x, y are already height-corrected; no correction needed |
| players | 20.5 visible on average, 22 on the median frame, 14 at the 5th percentile; slots are tracklets, not people | every player feature is a statistic over a set; head-counts are never used |
| second half | the graded block starts 2,933 frames before the kickoff, with a ball being passed about | needs an explicit kickoff rule (section 8) |

## 3. Approach

```
 10 public matches (other games) ──labels──┐
        │                                  │
        ▼                                  ▼
   degrade to look like the provided   H / A / D per frame
   tracking (section 5)                    │
        │                                  │
        ▼                                  ▼
   features: players · ball · visibility ──► LightGBM ×5:  alive?   which team?
                                                           team change · stoppage · restart (events)
 provided match
        │
        ▼
   same features ──► same five models ──► probabilities ──► kickoff rule ──► Viterbi (H/A/D) ──► CSV
```

Why this design rather than another:

- **Hand labels on the provided match are forbidden, and the public five minutes may not be
  trained or tuned on.** So everything that needs supervision must come from other games. What
  transfers between games is the generic part of the problem — how a team moves when the ball is
  dead, which player a ball has just left. What does not transfer is how *this* tracker behaves.
- So the tracker's behaviour is **measured without labels and imposed on the training data**,
  instead of being learned from labels. The alternative, a hand-built rule system, would have had
  every threshold chosen by eye.
- Alive/dead and team are **separate models with different inputs**, because the public sources
  have no usable ball during a stoppage (DFL freezes it, Metrica blanks it). The alive model is
  therefore kept away from fine ball kinematics, where it could only learn artefacts of the
  simulation; the team model is trained on live frames only, where the ball is real.

## 4. Training data

Every labelled frame comes from another match. Nothing was labelled by us by hand.

| source | matches | frames (25 Hz) | labels | who produced them | licence |
|---|---|---|---|---|---|
| DFL open dataset (Bassek et al., *Scientific Data* 2025) | 7 (Bundesliga, 2. Bundesliga) | 1,002,644 | `BallStatus` (in play / not) and `BallPossession` (home / away) on every ball frame | the league's data providers; shipped with the dataset | CC BY 4.0 |
| Metrica Sports sample data | 3 (anonymised) | 429,923 | derived by rule from the published events and ball track (below) | events by Metrica; derivation by `external/metrica_sample.py` | published for public use with attribution |

Total: 1,432,567 frames, 4,217 possessions, 1,057 stoppages. Each match is used twice, once with
the teams exchanged, so no model can prefer a side.

**Metrica labels.** These three games are the closest public thing to the challenge's ground
truth — corrected tracking with events, from the same provider — so the derivation mirrors the
organisers' description ("dead where the corrected tracking marks the ball dead, otherwise the team
it names as owning it"):

- *dead*: the corrected ball track is absent for at least 10 frames (Metrica blanks the ball out of
  play), **or** the stretch between the end of the last event and the restart that follows it (any
  `SET PIECE`, or a goal kick). The second clause catches stoppages where the ball was left
  visible, lying still: without it one sample game had a 161-second "possession".
- *owner*: the team of the most recent on-ball event (`PASS`, `CARRY`, `SHOT`, `RECOVERY`,
  `BALL LOST`, `SET PIECE`, `BALL OUT`). `CHALLENGE`, `FAULT RECEIVED` and `CARD` do not move it.
  After a stoppage the first event decides from the first live frame.

The two sources agree on what a possession is closely enough to be mixed: 3.8–5.3 possessions per
minute in DFL, 3.9–4.2 in Metrica; 9–17% and 12–14% of them shorter than one second. A model
trained on one source alone and tested on the other loses up to about three points of frame
accuracy against one trained on both.

## 5. Making public tracking look like the provided tracking

`degrade.py`. Each step is calibrated against a statistic of the provided match that needs no label.

| what | how | calibration target (provided match → degraded public) |
|---|---|---|
| ball smoothing | Gaussian, σ = 7 frames | kick pulse width 19 frames → 14–15 (1 before smoothing); velocity change over 0.5 s, percentiles 50/75/90/99: 3.2/5.8/8.4/14.7 → about 2.4/4.5/7.4/13.8 (DFL), 2.9/5.4/8.2/13.7 (Metrica) |
| ball position error | slow random offset, 0.25 m (DFL), 0.80 m (Metrica, whose ball is snapped onto the player at each event) | share of live frames with a player within 1 m of the ball: 0.29 → 0.26–0.28 (DFL), 0.21–0.26 (Metrica); 0.41 before degradation |
| ball visibility | **copied, not modelled**: the provided match's visible/missing masks are cut along its stoppages and live spells and pasted onto public stoppages and live spells of similar length, keeping both ends at true speed | by construction |
| stoppage ball | where a pasted mask says "visible" during a stoppage, a straight line from where play stopped to where it restarts | placeholder; hence the alive model's restricted inputs |
| players | random stretches of random tracks blanked | no model may rely on 22 complete trajectories |

**Where the stoppages of the provided match come from, without labels.** A players-only
alive/dead classifier (public matches, clean, no ball feature at all) is run on the provided match
and decoded into segments; the visibility masks are cut along those. The ball is deliberately not
consulted: the bank measures how visibility relates to stoppages, so the stoppages must come from a
view that shares nothing with it. We tried the obvious refinement — fitting both jointly with EM in
a two-state HMM — and it collapses to "missing = dead" (rates 0.001 / 0.999), which is circular and
wrong exactly where it matters. Re-cutting the bank along the final model's own output was also
tried and not adopted, for the same reason (it moves the median visible "head" of a stoppage from
30 frames to 1).

**Is the simulation realistic?** Two checks, neither used for calibration:

| organisers' baseline | simulated public matches (mean of 10) | published, private full match |
|---|---|---|
| raw | macroF1 0.651 · spellF1 0.089 | 0.653 · 0.049 |
| ± 25 frames | 0.662 · 0.227 | 0.675 · 0.200 |
| ± 50 frames | 0.652 · **0.248** | 0.672 · **0.243** |
| ± 125 frames | 0.593 · 0.210 | 0.636 · 0.224 |

and the rate at which the nearest player to the ball changes team: 22.3 per 1,000 visible live
frames in the provided match, 20.1–22.7 in the degraded public ones (24.3 before degradation).

The match is not reproduced exactly — the degraded balls are a little too far from the players and
turn a little too gently (table above) — and no further tuning of the simulation was done once
these two checks had been looked at, precisely so that they stay checks.

## 6. Features

`features.py`, 265 per frame. Each half is processed on its own and rotated so the home team
attacks towards +x. All windows are two-sided (0.25 s to 8 s).

- **Players (148, of which 138 used).** Per team: speed quantiles, share standing / sprinting,
  depth and width statistics, forward drift; across teams: gap between centroids, marking distance,
  players outside the lines; each with past, future, min and max windows. The ten head-count
  features are excluded: public tracking always has 22 players, the provided one does not.
- **Ball (110).** Distance to the nearest and second-nearest player of each team, soft "control"
  score, speed, velocity change, position relative to the lines; the same on windows before and
  after. Two families were added for the smoothed ball and carry most of the team model:
  - *de-smoothed position.* A low-pass filter rounds the corner of the trajectory at every kick,
    so at the instant of a kick the filtered ball is not at the kicker's feet. Extending the
    straight motion before the kick forwards, and the straight motion after it backwards, gives
    two estimates that meet at the kick point. At strong kicks this brings the median distance to
    the nearest player from 2.39 m to 1.35 m. The top team feature (23% of gain) is the control
    score computed on this position.
  - *touch chain.* Every local peak of velocity change, weak ones included, described by who was
    nearer the de-smoothed kick point, by how much, and how hard the ball was redirected — for
    the two touches before the frame and the two after.
- **Visibility (7).** Whether the track is missing, for how long it has been and will be.

The alive model sees players + visibility + the ball's position relative to the lines (148
columns); the team model sees everything except head-counts (255).

## 7. Models

Five LightGBM binary classifiers (500 trees for alive and team, 200 for each event detector, 63
leaves, learning rate 0.05, row and column subsampling, deterministic mode, fixed seed).

- **alive**: P(dead ball). 955k training rows (every third frame of 20 match-versions).
- **team**: P(home | ball alive), live frames only.
- **events**: P(a team change / a stoppage / a restart within ±2 frames).

Frames are weighted by (length of their possession)^-0.5, so that sixty one-second possessions are
not outvoted by one one-minute possession. That is worth 0.014 over unweighted training; equal
weight per possession (exponent 1) adds nothing further (section 10).

## 8. From probabilities to possessions

`decode.py`. A three-state HMM (H, A, D) solved exactly with Viterbi, each half on its own.

- Emissions are the classifiers' log-probabilities times a temper of 0.3, because consecutive
  frames are far from independent.
- Each kind of switch has its own price: team change 4.5, stoppage 14, restart 6.
- **Event-aware prices**: on each frame the price of a switch is lowered or raised by the log-ratio
  of the matching event detector to its average. A change of team is cheap on the frame where a
  touch by the other side was detected and dearer everywhere else.
- No minimum duration and no smoothing window. A short possession survives whenever its evidence
  outweighs the price of entering and leaving it.

The seven numbers were chosen by `spellF1` on held-out public matches (1,080 settings; the optimum
is flat — the top twelve are within 0.002).

**Opening kickoff** (`kickoff.py`). A half does not have to start at its kickoff, and to a model
people passing a ball about before it look like play. The public matches cannot teach the
difference (their recordings start at the whistle), so it is settled by what is true of every
kickoff: the teams stand either side of the halfway line and stop doing so about 1.3 s later. The
detector finds the first calm-then-mixed moment and steps back by the public-match median
crossing delay (32 frames, 48 kickoffs). It is applied only if the alive model itself calls the
first five seconds of the half a dead ball. On the public matches that happens in 5 of 20 halves
(Metrica recordings that start before the whistle): the detected kickoff is 0, 13, 21 and 31
frames late, and 426 frames late on a kickoff that was retaken; the validation score is unchanged
to four decimals. On the provided match it lands 1 frame and 13 frames from the anchors the
organisers publish in `games.py`; those anchors are used as a check only. Frames before the
detected kickoff are `D` (2,946 frames of the second half).

## 9. The possession rules this implements

| situation | what the system does |
|---|---|
| change of team | at the first touch of the team that takes over, as in the public ground truth; found from a change of ball velocity near a player of that team |
| a touch without control, a deflection | a possession of its own only if the evidence over those frames outweighs two team-change prices; in practice possessions under one second are rarely declared (7% recall) |
| loose or contested ball, aerial duel | stays with the previous team until the other side's touch is established — the HMM's persistence |
| ball track missing in live play | carried by the players-only features and the surrounding context; never forced to `D` |
| ball out of play, foul, goal | `D` from the alive model: players slow down, the track leaves the pitch or disappears |
| restart | alive again at the restart event; the team is whoever the ball leaves |
| before the opening kickoff of a half | `D` |
| outside the two graded halves | `X` (never scored). Inside them `X` is never used: for the ranking metric it would only repeat the previous label |

## 10. Validation and experiments

**Method.** Three folds over the ten public matches (each fold holds out one Metrica game and two
or three DFL games), held-out matches degraded like the provided one, scored with the organisers'
`possession_metrics.py`. Decoder settings are chosen on the pooled out-of-fold probabilities.
`metrics.py` is a fast copy of `spellF1` used inside tuning loops; a test checks it equals the
official scorer to 1e-12.

| # | system (all on held-out public matches) | spellF1 | macroF1 |
|---|---|---|---|
| 0 | organisers' baseline, ±50 frames | 0.248 | 0.652 |
| 1 | base features, GBDT alive + team, Viterbi | 0.637 | 0.908 |
| 2 | + de-smoothed ball position and touch chain | 0.651 | 0.912 |
| 3 | + event detectors and event-aware prices — **submitted** | **0.659** | **0.915** |
| 3a | same probabilities, per-frame argmax | 0.422 | 0.910 |
| 3b | same, argmax + majority vote ±12 / ±25 / ±50 | 0.617 / 0.638 / 0.602 | 0.911 / 0.911 / 0.908 |
| 4 | #3 with equal weight per possession (exponent 1) | 0.657 | 0.910 |
| 5 | #3 with unweighted training | 0.646 | 0.914 |
| 6 | hybrid: temporal convolutional network on the features and the tree models' probabilities — one fold, four matches | 0.667 | 0.926 |
| 6′ | #3 on those same four matches | 0.655 | 0.918 |

An earlier version with lighter ball smoothing (σ = 4) scored 0.667 on its own, easier, simulation;
it is not comparable with the rows above and is the reason the smoothing was re-measured.

Per held-out match, system #3: 0.621–0.706 (DFL), 0.631–0.668 (Metrica).

**Where the points are lost** (system #3):

| | spellF1 |
|---|---|
| as predicted | 0.659 |
| with a perfect alive/dead call | 0.711 |
| with a perfect team call | 0.938 |

| true possession length | share of possessions | recall at IoU 0.5 | precision of predictions that long |
|---|---|---|---|
| under 1 s | 13% | 0.07 | 0.44 |
| 1–2 s | 17% | 0.34 | 0.56 |
| 2–4 s | 22% | 0.55 | 0.67 |
| 4–10 s | 25% | 0.77 | 0.78 |
| over 10 s | 23% | 0.93 | 0.91 |

Of the missed possessions under four seconds, 44–55% are given to the other team (the touch was
not seen) and 33–43% have the right team but are merged with a neighbour because the short
possession *between* them was missed. Both are the same failure: short possessions. Boundary
timing was the other half of the story and is where the event detectors earn their place: at team
changes the median timing error of matched possessions fell from 11 frames to 4 (53% within 5
frames), though a quarter are still more than 32 frames out.

**The public five minutes.** Used as the rules intend: to check the pipeline. The submitted file
scores spellF1 0.736, macroF1 0.929 on it (14 possessions predicted, 17 true; alive/dead F1 0.98;
median transition latency −2.5 frames, so there is no alignment offset). With 17 possessions the
scorer's own bootstrap interval is 0.53–0.89; nothing was selected on this number.

## 11. Assets: used, tried, dropped

| asset | status | why |
|---|---|---|
| tactical tracking (ATD) | **used** | the only input of the submitted system |
| homography | examined, not used | it gave the camera pose (and through it the proof that airborne balls are already height-corrected); the camera's view centre follows the players (r = 0.96) more than the ball (r = 0.83), so it adds nothing the tracking lacks |
| Smart Tagging | examined, not used | clips are fixed-length and padded (set pieces 35.0 s, several phases exactly 30.0 s); "build up" and "progression" already agree with our team call on 93% and 89% of live frames, the other phases on 62–76%. A coarse prior of that kind can only cost short possessions, and its weight could not be validated without labels |
| tactical video, broadcast video and tracking | not used, not examined | the approach works from the tactical tracking alone; no GPU was available to process video, and the broadcast feed has no mapping to tactical frames |
| public five-minute ground truth | pipeline check only | as the rules require |
| `games.py` kickoff anchors | check only | the kickoff is detected from tracking instead |

Modelling ideas tried and not adopted: joint EM of visibility and player movement (collapses);
re-estimating the visibility bank from the model's own output (circular); equal weight per
possession (no gain); majority-vote smoothing (2 points worse than Viterbi).

**The neural network.** A non-causal dilated convolutional network (7 residual blocks, 64
channels, ±10 s receptive field) reading the engineered features together with the trees'
out-of-fold probabilities was trained on six public matches and tested on the other four (rows 6
and 6′; both decoders tuned on those four). It is better on all four, by 0.012 spellF1 and 0.009
macroF1. That is inside the metric's stated noise band of 0.04, and adopting it would have put a
component whose output is not guaranteed bit-identical across machines into a pipeline that must
regenerate the same file. It was left out of the submission and is the first thing to pick up
with more time.

## 12. Limitations

- **Short possessions.** Three in ten possessions last under two seconds and we recover about a
  fifth of them. With a ball track smoothed over 0.8 s we believe this is close to what tracking
  alone can give; detecting the ball in the video would be the way past it.
- **The stoppage ball is simulated.** No public source records a real ball during a stoppage, so
  the alive model cannot use what the provided track does while play is stopped. A perfect
  alive/dead call would be worth about five points.
- **Validation is on simulated degradation.** The simulation comes close to the provided match on
  the label-free statistics we measured and on the organisers' baseline, but not exactly (section
  5), the upper quantiles of ball speed in the provided match are still 10–25% higher than in the
  public ones, and one match cannot tell us how the system behaves on another tracker.
- **Label conventions.** DFL and Metrica do not define possession identically; the challenge's
  ground truth follows Metrica's. We mix both for volume.
- **The kickoff detector** assumes a half opens either in play or with a visible line-up.
- Expect a lower score on the private ground truth than on our validation: the public matches are
  cleaner than any simulation of noise can make them.

## 13. Reproduction

Python 3.10, pinned in `requirements.txt` (numpy 1.24.3, scipy 1.15.3, pandas 2.2.3, lightgbm
4.5.0, scikit-learn 1.5.0, joblib 1.4.2).

```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data PYTHONPATH=src
python scripts/predict.py                       # committed models -> evaluation/results/wa_barca.csv
python evaluation/validate_submission.py evaluation/results/wa_barca.csv
```

From scratch: `prepare_external.py` (downloads the public matches), `bootstrap.py`, `train.py`,
`predict.py`; validation with `run_cv.py`, `tune_decoder.py`, `error_analysis.py`,
`ablate_decoder.py`, `baseline_check.py`, `domain_check.py`. Seeds are fixed, LightGBM runs in
deterministic mode, and the whole chain was re-run to confirm it regenerates the same file
(section "Reproducibility check" of `experiments/README.md`). Runtime on 8 CPU cores, no GPU:
features 50 s for the twenty training tables, training 7 minutes, inference 30 seconds.
