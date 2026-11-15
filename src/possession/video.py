"""Ball detection in the tactical video. An experiment, not part of the submitted pipeline.

Why. The error analysis says the remaining points are in short possessions, and that they are
limited by the ATD ball itself: smoothed over ~0.8 s and missing on 41% of graded frames. A ball
detected frame by frame in the video could be sharper and fill some of the gaps.

What this module answers, without any label:
  1. agreement   where the ATD has a ball, is the detected ball in the same place?
                 (if not, the detector is finding heads and shoes, or the frames are misaligned)
  2. gain        where the ATD has NO ball but the players are moving (a label-free proxy for live
                 play), how often does the video give one?
  3. cost        how long a full run takes.

The detector is pretrained on COCO (other images; class "sports ball"), so nothing is trained or
labelled on the provided match. Detection runs on 640x640 crops at native resolution, centred on
where the ball is expected (the ATD ball projected through the homography, else the last
detection), because the ball is only a few pixels wide in the wide tactical view and shrinking the
whole frame to 640 would erase it. With no expectation at all the whole frame is tiled, but only
every few frames, to re-acquire the ball cheaply.

Everything except `read_frames` and `yolo_detector` is plain numpy, so it is tested without a
video (tests/test_video.py). Outputs are derived match data: keep them in the data folder's
.cache/ and never commit them.
"""
import time
import warnings

import numpy as np

SPORTS_BALL = 32           # COCO class id
CROP = 640                 # detector input, native pixels
OVERLAP = 64               # between whole-frame tiles
MAX_JUMP = 150.0           # px; a candidate further than this from the expected position ...
SURE = 0.5                 # ... is kept only if the detector is at least this confident
RECENT = 25                # frames a previous detection stays usable as the expected position
FULL_SEARCH_EVERY = 5      # with no expected position, tile the whole frame only every k-th frame

MODE_SKIPPED, MODE_ATD, MODE_LAST, MODE_FULL = 0, 1, 2, 3


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------

def _apply(M, x, y):
    w = M[:, 2, 0] * x + M[:, 2, 1] * y + M[:, 2, 2]
    return np.stack([(M[:, 0, 0] * x + M[:, 0, 1] * y + M[:, 0, 2]) / w,
                     (M[:, 1, 0] * x + M[:, 1, 1] * y + M[:, 1, 2]) / w], 1)


def project(H, xy_norm):
    """Pitch (0-1) -> pixels, frame by frame. H [n,3,3], xy_norm [n,2]; NaN in, NaN out."""
    return _apply(H, xy_norm[:, 0], xy_norm[:, 1])


def unproject(H, uv):
    """Pixels -> pitch (0-1), frame by frame. A ball in the air lands where it would on the ground."""
    ok = np.isfinite(H).all((1, 2))
    Hinv = np.full_like(H, np.nan)
    Hinv[ok] = np.linalg.inv(H[ok])
    return _apply(Hinv, uv[:, 0], uv[:, 1])


def tile_origins(width, height, size=CROP, overlap=OVERLAP):
    """Top-left corners of size x size tiles covering the whole frame."""
    def starts(total):
        if total <= size:
            return [0]
        return list(range(0, total - size, size - overlap)) + [total - size]
    return [(x, y) for y in starts(height) for x in starts(width)]


def crop_origin(u, v, width, height, size=CROP):
    """Top-left corner of the size x size crop centred on (u, v), kept inside the frame."""
    x = int(round(u - size / 2.0))
    y = int(round(v - size / 2.0))
    return min(max(x, 0), max(width - size, 0)), min(max(y, 0), max(height - size, 0))


# ----------------------------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------------------------

def pick(cands, expected=None):
    """The ball among candidates [k, 3] (u, v, conf) in frame pixels, or None.

    Near the expected position the most confident candidate wins; far from it only a confident
    one is believed (a ball can move 150 px in a frame only on a fast shot seen up close).
    """
    cands = np.asarray(cands, np.float64).reshape(-1, 3)
    if expected is not None and len(cands):
        d = np.hypot(cands[:, 0] - expected[0], cands[:, 1] - expected[1])
        cands = cands[(d <= MAX_JUMP) | (cands[:, 2] >= SURE)]
    if not len(cands):
        return None
    return cands[int(np.argmax(cands[:, 2]))]


def new_state(n):
    """Per-frame results; updated in place by detect(), so a run can be saved and resumed."""
    return {"uv": np.full((n, 2), np.nan, np.float32), "conf": np.full(n, np.nan, np.float32),
            "mode": np.zeros(n, np.int8), "done": np.zeros(n, bool)}


def detect(frames, size, expected_px, detector, state=None, batch=16, every=FULL_SEARCH_EVERY):
    """Run `detector` over `frames` and fill `state`.

    frames       iterable of (frame index, image [h, w, 3]), increasing indices
    size         (width, height) of the video
    expected_px  [n, 2] where the ATD puts the ball in pixels, NaN where it has none
    detector     callable: list of crops -> list of [k, 3] arrays (u, v, conf) in crop pixels
    """
    width, height = size
    st = new_state(len(expected_px)) if state is None else state
    last = {"uv": None, "frame": -10 ** 9}
    queue = []

    def expectation(f):
        if np.isfinite(expected_px[f]).all():
            return expected_px[f], MODE_ATD
        if last["uv"] is not None and f - last["frame"] <= RECENT:
            return last["uv"], MODE_LAST
        return None, MODE_FULL

    def flush():
        crops = [c for _, _, _, cs in queue for c in cs]
        outs = detector(crops) if crops else []
        k = 0
        for f, mode, origins, _ in queue:
            found = [np.asarray(outs[k + i], np.float64).reshape(-1, 3) + (x0, y0, 0.0)
                     for i, (x0, y0) in enumerate(origins)]
            k += len(origins)
            exp, _ = expectation(f)
            hit = pick(np.concatenate(found), exp)
            st["mode"][f], st["done"][f] = mode, True
            if hit is not None:
                st["uv"][f], st["conf"][f] = hit[:2], hit[2]
                last["uv"], last["frame"] = hit[:2], f
        queue.clear()

    n_crops = 0
    for f, img in frames:
        if st["done"][f]:
            continue
        centre, mode = expectation(f)
        if centre is not None:
            origins = [crop_origin(centre[0], centre[1], width, height)]
        elif f % every == 0:
            origins = tile_origins(width, height)
        else:
            st["mode"][f], st["done"][f] = MODE_SKIPPED, True
            continue
        queue.append((f, mode, origins, [img[y:y + CROP, x:x + CROP] for x, y in origins]))
        n_crops += len(origins)
        if n_crops >= batch:
            flush()
            n_crops = 0
    flush()
    return st


# ----------------------------------------------------------------------------------------------
# what needs a video and a GPU (imported lazily: the rest of the module runs without them)
# ----------------------------------------------------------------------------------------------

def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open {}".format(path))
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frames(path, wanted, seek_gap=250):
    """Yield (frame index, BGR image) for the sorted frame indices `wanted`.

    Reads sequentially and seeks only across long jumps. Seeking in compressed video can land on
    the wrong frame with some codecs: notebooks/video_ball_test.ipynb checks that it does not.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    pos = 0
    try:
        for f in np.asarray(wanted, np.int64):
            if f - pos > seek_gap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                pos = int(f)
            while pos < f:
                cap.grab()
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                return
            yield int(f), img
    finally:
        cap.release()


def best_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def yolo_detector(weights="yolov8m.pt", device=None, conf=0.05, imgsz=CROP):
    """A detector callable for detect(): COCO-pretrained YOLO, class "sports ball" only."""
    from ultralytics import YOLO
    device = device or best_device()
    model = YOLO(weights)

    def run(crops):
        res = model.predict(crops, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL], device=device, verbose=False)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 3)))
                continue
            b = r.boxes.xyxy.cpu().numpy()
            out.append(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, r.boxes.conf.cpu().numpy()], 1))
        return out

    run.device = device
    run.weights = weights
    return run


# ----------------------------------------------------------------------------------------------
# label-free report
# ----------------------------------------------------------------------------------------------

def near_player_share(ball_m, players_m, mask):
    """Share of `mask` frames with a player within 1 m of the ball. A sharp, well-placed ball sits
    at a player's feet more often: 0.29 for the provided ATD ball, 0.41 for clean public tracking."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.nanmin(np.hypot(players_m[mask, :, 0] - ball_m[mask, None, 0],
                               players_m[mask, :, 1] - ball_m[mask, None, 1]), 1)
    return float(np.mean(d < 1.0)) if mask.any() else None


def report(st, H, atd_ball_m, pitch, period, players_speed, players_m=None, frames=None):
    """Aggregate, label-free comparison of the video ball with the ATD ball, over processed graded
    frames (or over `frames`, a bool mask). Only numbers: safe to share and commit."""
    scope = st["done"] & (period > 0) if frames is None else st["done"] & frames
    det = np.isfinite(st["uv"][:, 0]) & scope
    atd = np.isfinite(atd_ball_m[:, 0]) & scope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moving = np.nanmedian(players_speed, 1) > 1.5
    xy = unproject(H, st["uv"].astype(np.float64)) * np.asarray(pitch, np.float64)
    d = np.hypot(*(xy - atd_ball_m).T)
    both = det & atd
    gap_live = scope & ~atd & moving

    def share(a, b):
        return float(a.sum()) / max(int(b.sum()), 1)

    r = {
        "frames processed": int(scope.sum()),
        "detection rate, all processed frames": share(det, scope),
        "detection rate where the ATD has a ball": share(det & atd, atd),
        "detection rate where the ATD has none": share(det & ~atd, scope & ~atd),
        "detection rate where the ATD has none and players move (live-play proxy)": share(det & gap_live, gap_live),
        "share of processed frames that are ATD gaps with players moving": share(gap_live, scope),
        "distance video vs ATD (m), median": float(np.nanmedian(d[both])) if both.any() else None,
        "distance video vs ATD (m), 90th percentile": float(np.nanpercentile(d[both], 90)) if both.any() else None,
        "share of agreeing frames within 1 m": share(both & (d < 1.0), both),
        "share of agreeing frames further than 5 m (false or aerial detections)": share(both & (d > 5.0), both),
        "frames by search mode (skipped, ATD crop, last-detection crop, full frame)":
            [int(((st["mode"] == m) & scope).sum()) for m in range(4)],
    }
    if players_m is not None:
        r["player within 1 m of the ball, ATD ball (frames where both exist)"] = near_player_share(atd_ball_m, players_m, both)
        r["player within 1 m of the ball, video ball (same frames)"] = near_player_share(xy, players_m, both)
    return r


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def eta(self, done, total):
        el = time.time() - self.t0
        rate = done / el if el > 0 else 0.0
        return rate, (total - done) / rate if rate > 0 else float("inf")
