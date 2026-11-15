# Calibration notes

How the degradation in `src/possession/degrade.py` was set, and the facts about the provided
tracking the design rests on. Everything on this page is measured **without labels**: "live"
frames of the provided match are the ones a players-only model calls live.

## Output of `python scripts/calibrate.py`

```
PROVIDED MATCH
  graded frames 158857; ball track present on 59.3%
  149 visible runs, 150 gaps; median gap 10.7 s; 99.9% of missing frames are in gaps of 1 s or more
  median player speed: 0.97 m/s while the ball track is missing, 1.62 m/s while present; 87% of missing frames are under 1.5 m/s
  visible players per graded frame: mean 20.5, median 22, 5th percentile 14
  lateral acceleration of balls faster than 14 m/s: mean -0.44 m/s^2 over 9695 frames (an uncorrected projection from this camera would give about +7)
  bank[dead]: 99 segments; ball visible on 26.1% of their frames; 16% never visible, 21% always
  bank[alive]: 98 segments; ball visible on 89.2% of their frames; 0% never visible, 33% always

CALIBRATION, live frames with a visible ball
  pulse width (frames)
      provided match                   19
      dfl_J03WN1 (degraded)            14
      dfl_J03WQQ (degraded)            14
      metrica_sample_1 (degraded)      15
      metrica_sample_3 (degraded)      15
      dfl_J03WQQ (as published)        1
  velocity change over 0.5 s, pct 50/75/90/99
      provided match                   [3.2, 5.8, 8.4, 14.7]
      dfl_J03WN1 (degraded)            [2.2, 4.3, 7.3, 13.9]
      dfl_J03WQQ (degraded)            [2.5, 4.7, 7.5, 13.6]
      metrica_sample_1 (degraded)      [2.9, 5.4, 8.2, 14.4]
      metrica_sample_3 (degraded)      [2.9, 5.4, 8.1, 12.9]
      dfl_J03WQQ (as published)        [2.7, 5.5, 12.3, 25.9]
  share with a player within 1 m
      provided match                   0.292
      dfl_J03WN1 (degraded)            0.284
      dfl_J03WQQ (degraded)            0.262
      metrica_sample_1 (degraded)      0.263
      metrica_sample_3 (degraded)      0.212
      dfl_J03WQQ (as published)        0.407
  ball speed, pct 50/75/90/99
      provided match                   [6.0, 10.4, 14.7, 23.8]
      dfl_J03WN1 (degraded)            [5.7, 9.3, 14.1, 21.0]
      dfl_J03WQQ (degraded)            [5.9, 9.8, 13.9, 20.0]
      metrica_sample_1 (degraded)      [6.3, 10.2, 14.1, 20.0]
      metrica_sample_3 (degraded)      [6.1, 10.0, 13.3, 18.5]
      dfl_J03WQQ (as published)        [6.1, 11.0, 15.6, 23.8]
  nearest-team switches per 1000 frames
      provided match                   22.3
      dfl_J03WN1 (degraded)            22.7
      dfl_J03WQQ (degraded)            20.1
      metrica_sample_1 (degraded)      22.3
      metrica_sample_3 (degraded)      22.4
      dfl_J03WQQ (as published)        24.3
```

## How the constants were chosen

**Ball smoothing, σ = 7 frames.** A kick is a step in velocity; a Gaussian filter of width σ turns
it into an acceleration pulse of standard deviation σ. The mean pulse of the provided match,
normalised to its peak, is 0.79 / 0.89 / 0.96 at ±6 / ±4 / ±2 frames, which is the profile of
σ ≈ 8.5. A grid over σ and the positional offset was then compared on three statistics at once
(one DFL match and one Metrica game; live frames, before visibility loss and track loss):

| σ | offset | players within 1 m | velocity change, pct 50/75/90/99 | pulse width |
|---|---|---|---|---|
| provided match | | 0.291 | 3.0 / 5.6 / 8.3 / 14.6 | 19 |
| DFL 6.0 | 0.25 m | 0.312 | 2.6 / 4.9 / 8.4 / 16.2 | 14 |
| DFL **7.0** | **0.25 m** | 0.296 | 2.5 / 4.8 / 7.7 / 14.5 | 16 |
| DFL 8.5 | 0.25 m | 0.271 | 2.5 / 4.4 / 6.8 / 12.3 | 18 |
| Metrica 6.0 | 0.80 m | 0.304 | 2.9 / 5.7 / 8.9 / 15.7 | 13 |
| Metrica **7.0** | **0.80 m** | 0.295 | 2.9 / 5.4 / 8.2 / 14.1 | 15 |
| Metrica 8.5 | 0.80 m | 0.279 | 2.9 / 5.0 / 7.3 / 12.1 | 17 |

σ = 8.5 matches the pulse width best and the velocity changes worst; σ = 6 the reverse. σ = 7 was
taken as the compromise, with the offset set per source to match the share of frames with a
player within a metre. The first version of the pipeline used σ = 4, chosen on acceleration
percentiles alone; the pulse width showed it was far too light.

The full degradation (first block above) also blanks player tracks and hides the ball, which
pushes the proximity share a little below the provided match's. That residual was left alone.

**Positional offset correlation, 12 frames.** Not calibrated; a detector's error is slow compared
with the frame rate, and 0.5 s was taken as a plain assumption.

**Track loss.** Each track is blanked at random about a tenth of the time, in stretches of 6 s on
average, against 1.5 of 22 players missing per frame in the provided match. This is an
augmentation, not a model of which players a camera loses.

**Kickoff crossing delay, 32 frames.** Median over the 48 kickoffs of the ten public matches of
the time from the restart to the third player crossing the halfway line (interquartile range
21–41 frames).

## Facts established along the way

- **The camera.** With square pixels and the principal point at the image centre, the homography
  decomposes into a focal length of about 2,350 px and a camera centre 54 m along the pitch, 45 m
  behind the touchline at y = 68 m and 30 m above it (interquartile range of the estimate over the
  match: 5 m, 13 m, 1 m). The image centre, projected onto the pitch, follows the mean player
  position (r = 0.96) more closely than the ball (r = 0.83).
- **Airborne balls are not displaced.** From that camera a ball 3 m up at the centre of the pitch would be drawn about 9 m further
  away, and a flight would bulge away from the camera with an apparent lateral acceleration of
  about +7 m/s² on average for fast balls (+16 at the median of the fastest; measured by applying
  that projection to a DFL match, which records height). The provided ball shows −0.4. Its x, y
  are therefore already corrected for height, and no correction is applied.
- **The ball track is switched off, not lost.** 150 gaps with a median of 10.7 s, during which the
  players are mostly walking; but 21% of the stoppages the players-only model finds have the ball
  visible throughout, and about a tenth of live frames have none.
