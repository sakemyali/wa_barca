"""The video experiment's logic, without a video: synthetic frames and a fake detector."""
import numpy as np

from possession import video as vid


def test_tiles_cover_frame():
    origins = vid.tile_origins(1920, 1080)
    cover = np.zeros((1080, 1920), bool)
    for x, y in origins:
        cover[y:y + vid.CROP, x:x + vid.CROP] = True
    assert cover.all() and len(origins) == 8


def test_crop_stays_inside():
    assert vid.crop_origin(5, 5, 1920, 1080) == (0, 0)
    assert vid.crop_origin(1915, 1075, 1920, 1080) == (1920 - 640, 1080 - 640)
    assert vid.crop_origin(960, 540, 1920, 1080) == (640, 220)


def test_project_unproject_round_trip():
    rng = np.random.default_rng(0)
    H = np.tile(np.array([[1800.0, -500, 40], [2, 170, 250], [-0.02, -0.5, 1]]), (50, 1, 1))
    H += rng.normal(0, 0.001, H.shape)
    xy = rng.uniform(0.05, 0.95, (50, 2))
    back = vid.unproject(H, vid.project(H, xy))
    assert np.allclose(back, xy, atol=1e-9)
    H[3] = np.nan
    assert np.isnan(vid.unproject(H, vid.project(H, xy))[3]).all()


def test_pick_prefers_near_unless_sure():
    c = np.array([[100, 100, 0.3], [600, 600, 0.4]])
    assert tuple(vid.pick(c, (110, 100))[:2]) == (100, 100)        # far one is dropped
    c2 = np.array([[100, 100, 0.3], [600, 600, 0.9]])
    assert tuple(vid.pick(c2, (110, 100))[:2]) == (600, 600)       # unless confident
    assert vid.pick(np.zeros((0, 3)), None) is None


def _fake_video(n, width=1920, height=1080, ball=None):
    """Frames that are black except for a white 6x6 square at ball[f]; ball NaN = not shown."""
    for f in range(n):
        img = np.zeros((height, width, 3), np.uint8)
        if ball is not None and np.isfinite(ball[f]).all():
            u, v = ball[f].astype(int)
            img[v - 3:v + 3, u - 3:u + 3] = 255
        yield f, img


def _fake_detector(crops):
    out = []
    for c in crops:
        ys, xs = np.nonzero(c[:, :, 0])
        out.append(np.array([[xs.mean() + 0.5, ys.mean() + 0.5, 0.9]]) if len(xs) else np.zeros((0, 3)))
    return out


def test_detect_follows_ball_and_reacquires():
    n = 60
    t = np.arange(n)
    ball = np.stack([300 + 20 * t, 500 + 5 * t], 1).astype(float)
    ball[40:45] = np.nan                                     # not visible in the video
    expected = ball.copy()
    expected[10:30] = np.nan                                 # ATD gap: must use last detection
    expected[50:] = np.nan
    calls = []

    def det(crops):
        calls.append(len(crops))
        return _fake_detector(crops)

    st = vid.detect(_fake_video(n, ball=ball), (1920, 1080), expected, det, batch=4)
    seen = np.isfinite(ball[:, 0])
    assert st["done"].all()
    assert np.allclose(st["uv"][seen & np.isfinite(st["uv"][:, 0])], ball[seen & np.isfinite(st["uv"][:, 0])], atol=1)
    assert np.isfinite(st["uv"][:40, 0]).all()                # including the ATD gap 10..29
    assert (st["mode"][10:30] == vid.MODE_LAST).all()
    assert np.isnan(st["uv"][40:45, 0]).all()
    # after the gap 40..44 the last detection is still recent, so 45.. is found again
    assert np.isfinite(st["uv"][45:, 0]).all()


def test_detect_full_search_only_every_k_frames():
    n = 20
    expected = np.full((n, 2), np.nan)
    st = vid.detect(_fake_video(n), (1920, 1080), expected, _fake_detector, every=5)
    assert (st["mode"][::5] == vid.MODE_FULL).all()
    assert (st["mode"][np.arange(n) % 5 != 0] == vid.MODE_SKIPPED).all()


def test_detect_resumes():
    n = 10
    ball = np.tile([[800.0, 400.0]], (n, 1))
    st = vid.detect(_fake_video(5, ball=ball), (1920, 1080), ball, _fake_detector)
    assert st["done"][:5].all() and not st["done"][5:].any()
    st = vid.detect(_fake_video(n, ball=ball), (1920, 1080), ball, _fake_detector, state=st)
    assert st["done"].all() and np.isfinite(st["uv"][:, 0]).all()


def test_report_agreement():
    n = 100
    pitch = (105.0, 68.0)
    H = np.tile(np.array([[1000.0, 0, 100], [0, 600, 100], [0, 0, 1]]), (n, 1, 1))
    ball_norm = np.tile([[0.5, 0.5]], (n, 1))
    st = vid.new_state(n)
    st["done"][:] = True
    st["uv"][:50] = vid.project(H, ball_norm)[:50]
    atd = ball_norm * pitch
    atd[25:75] = np.nan
    players = np.tile(np.array([[[52.5, 34.5], [10.0, 10.0]]]), (n, 1, 1))     # one player 0.5 m away
    r = vid.report(st, H, atd, pitch, np.ones(n, np.int8), np.full((n, 3), 3.0), players)
    assert r["distance video vs ATD (m), median"] < 1e-6
    assert r["detection rate where the ATD has a ball"] == 0.5
    assert r["detection rate where the ATD has none and players move (live-play proxy)"] == 0.5
    assert r["player within 1 m of the ball, video ball (same frames)"] == 1.0
