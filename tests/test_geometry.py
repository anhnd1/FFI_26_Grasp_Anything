"""Geometry/format sanity checks. Run: python tests/test_geometry.py  (or pytest)"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.formats import FORMATS  # noqa: E402
from grasp.geometry import (angle_diff, from_endpoints, iou, is_success, medoid_index,  # noqa: E402
                            to_endpoints)


def close(g1, g2, tol=1e-6):
    return all(abs(a - b) < tol for a, b in zip(g1[:4], g2[:4])) and angle_diff(g1[4], g2[4]) < tol


def test_endpoint_roundtrip():
    rng = random.Random(0)
    for _ in range(1000):
        g = (rng.uniform(0, 416), rng.uniform(0, 416), rng.uniform(5, 250), rng.uniform(2, 90), rng.uniform(0, 180))
        assert close(from_endpoints(*to_endpoints(*g)), g)


def test_endpoint_order_canonical():
    # theta and theta+180 are the same grasp -> identical endpoints
    (a1, a2, _), (b1, b2, _) = to_endpoints(100, 100, 50, 20, 30), to_endpoints(100, 100, 50, 20, 210)
    assert math.dist(a1, b1) < 1e-9 and math.dist(a2, b2) < 1e-9


def test_iou():
    g = (200, 200, 100, 40, 30)
    assert abs(iou(g, g) - 1) < 1e-9
    assert abs(iou(g, (200, 200, 100, 40, 210)) - 1) < 1e-9
    assert iou(g, (10, 10, 10, 10, 0)) == 0
    a, b = (0, 0, 2, 2, 0), (1, 0, 2, 2, 0)  # half overlap: 2 / (4 + 4 - 2)
    assert abs(iou(a, b) - 2 / 6) < 1e-9
    assert abs(iou((200, 200, 100, 40, 0), (200, 200, 100, 40, 90)) - (40 * 40) / (2 * 4000 - 1600)) < 1e-9


def test_angle_and_success():
    assert angle_diff(179, 1) == 2
    g = (200, 200, 100, 40, 10)
    assert is_success((205, 200, 100, 40, 170), [g])        # 20 deg off across the wrap
    assert not is_success((200, 200, 100, 40, 50), [g])     # 40 deg off
    assert is_success((200, 200, 100, 40, 50), [(0, 0, 5, 5, 0), (200, 200, 100, 40, 45)])  # any-GT match


def test_medoid():
    gs = [(100, 100, 50, 20, 0), (102, 100, 50, 20, 0), (104, 100, 50, 20, 0), (300, 300, 50, 20, 90)]
    assert medoid_index(gs) in (1,)


def test_formats_roundtrip():
    rng = random.Random(1)
    for name, fmt in FORMATS.items():
        for _ in range(200):
            g = [rng.uniform(20, 396), rng.uniform(20, 396), rng.uniform(20, 200), rng.uniform(5, 60), rng.uniform(0, 180)]
            text = fmt.encode(g, [g])
            back = fmt.decode(text)
            assert back is not None, (name, text)
            # integer 0-1000 quantization => within ~0.5 px per coordinate; angle within ~1 deg for long jaws
            assert math.dist(back[:2], g[:2]) < 1.5 and abs(back[2] - g[2]) < 1.5 and abs(back[3] - g[3]) < 1.5, (name, g, back)
            assert angle_diff(back[4], g[4]) < 1.5, (name, g, back)


def test_decode_trailing_text_and_digit_keys():
    # real failure from epbox run: JSON followed by runaway generation; fallback must skip the "2" in "bbox_2d"
    out = '{"bbox_2d": [0, 111, 90, 161], "points_2d": [[-83, 140], [89, 132]], "jaw": 43}\nuserGrip hairbrush on'
    ref = FORMATS["endpoints"].decode('{"points_2d": [[-83, 140], [89, 132]], "jaw": 43}')
    assert FORMATS["endpoints_box"].decode(out) == ref
    broken = 'bbox_2d: [0, 111, 90, 161] points_2d: [[-83, 140], [89, 132]] jaw: 43'  # not JSON -> number fallback
    assert FORMATS["endpoints_box"].decode(broken) == ref
    assert FORMATS["endpoints"].decode('points_2d: [[-83, 140], [89, 132]] jaw: 43') == ref


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
