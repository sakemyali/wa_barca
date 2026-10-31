import os

import numpy as np
import pytest

from possession import atd

RAW = "1:0.5,0.5,1.0;NaN,NaN,NaN:0.25,0.75,2.0\n2:NaN,NaN,NaN;0.1,0.2,3.0:NaN,NaN,NaN\n3:0.0,1.0,0.5;NaN,NaN,NaN:0.5,0.5,0.0\n"

META = """<main><Metadata><Teams>
<Team id="h"><Name>Home FC</Name></Team><Team id="a"><Name>Away FC</Name></Team>
</Teams><Players><Player id="1" teamId="h"/><Player id="2" teamId="a"/></Players></Metadata></main>"""


def test_parse_shapes_nan_and_offset():
    ids, ball, pxy, spd = atd.parse_atd(RAW, 2)
    assert ids.tolist() == [1, 2, 3]
    assert ball.shape == (3, 2) and pxy.shape == (3, 2, 2) and spd.shape == (3, 2)
    assert np.isnan(ball[1]).all() and ball[0].tolist() == [0.25, 0.75]
    assert np.isnan(pxy[0, 1]).all() and np.isnan(spd[2, 1])
    assert pxy[1, 1].tolist() == pytest.approx([0.1, 0.2])
    d = atd._to_atd(ids, ball, pxy, spd, np.array([0, 1], dtype=np.int8), "H", "A", (100.0, 50.0))
    assert d.frame.tolist() == [0, 1, 2]  # video_frame = frame_id - 1
    assert d.ball_xy.dtype == np.float32 and d.ball_xy[0].tolist() == [25.0, 37.5]
    assert d.n_frames == 3


def test_team_mapping(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(META)
    team, home, away = atd.parse_meta(p)
    assert team.tolist() == [0, 1] and team.dtype == np.int8
    assert (home, away) == ("Home FC", "Away FC")


@pytest.mark.skipif(not os.environ.get("MTAH_DATA_DIR"), reason="MTAH_DATA_DIR unset")
def test_real_tactical():
    d = atd.load_atd("tactical")
    assert d.n_frames == 199319
    assert d.players_xy.shape == (199319, 46, 2)
    assert 56 <= atd.check_alignment(d) <= 58
