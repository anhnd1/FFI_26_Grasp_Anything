"""Text formats for emitting a grasp from the VLM's LM head.

All formats use Qwen's relative grounding frame: integers in [0, 1000] (x / 416 * 1000), so the
baseline and ours differ only in *what* is written, not in the coordinate frame.

  xywht          baseline: center, size, angle in degrees
  endpoints      ours: two jaw endpoints (canonical order) + jaw size, i.e. points the VLM already knows how to emit
  endpoints_box  ours + box-first: target box (bbox of all GT grasps of the sample), then endpoints

decode() is tolerant: it parses the first JSON object in the text (generation may run on past it), then falls
back to the numbers in order. It returns (x, y, w, h, theta) in pixels, or None if the text cannot be parsed.
"""
import json
import re

from .geometry import bbox, from_endpoints, from_norm, norm_theta, to_endpoints, to_norm

# not preceded by a word char, so the "2" in keys like "bbox_2d" / "points_2d" is not a number
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _q(v):
    return int(round(to_norm(v)))


def _numbers(text, n, skip=0):
    nums = [float(v) for v in _NUM.findall(text)]
    return nums[skip:skip + n] if len(nums) >= skip + n else None


def _first_json(text):
    """First JSON object in text, ignoring anything generated after it."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    return json.JSONDecoder().raw_decode(text[start:])[0]


class XYWHT:
    name = "xywht"
    instruction = ('Output the grasp rectangle as JSON {"center": [x, y], "size": [w, h], "angle": a}, '
                   'with coordinates in [0, 1000] and the angle in degrees [0, 180).')

    def encode(self, g, gts):
        x, y, w, h, t = g
        return json.dumps({"center": [_q(x), _q(y)], "size": [_q(w), _q(h)], "angle": int(round(norm_theta(t))) % 180})

    def decode(self, text):
        try:
            d = _first_json(text)
            v = [*d["center"], *d["size"], d["angle"]]
        except (ValueError, KeyError, TypeError):
            v = _numbers(text, 5)
        if v is None or len(v) != 5:
            return None
        x, y, w, h, t = map(float, v)
        return (from_norm(x), from_norm(y), from_norm(w), from_norm(h), norm_theta(t))


class Endpoints:
    name = "endpoints"
    instruction = ('Output the two gripper jaw points and the jaw size as JSON '
                   '{"points_2d": [[x1, y1], [x2, y2]], "jaw": s}, with coordinates in [0, 1000].')

    def encode(self, g, gts):
        p1, p2, h = to_endpoints(*g)
        return json.dumps({"points_2d": [[_q(p1[0]), _q(p1[1])], [_q(p2[0]), _q(p2[1])]], "jaw": _q(h)})

    def _skip(self):
        return 0

    def decode(self, text):
        try:
            d = _first_json(text)
            (x1, y1), (x2, y2) = d["points_2d"]
            v = [x1, y1, x2, y2, d["jaw"]]
        except (ValueError, KeyError, TypeError):
            v = _numbers(text, 5, self._skip())
        if v is None or len(v) != 5:
            return None
        x1, y1, x2, y2, s = (from_norm(float(a)) for a in v)
        return from_endpoints((x1, y1), (x2, y2), s)


class EndpointsBox(Endpoints):
    name = "endpoints_box"
    instruction = ('First output the box of the target part, then the two gripper jaw points and the jaw size, as JSON '
                   '{"bbox_2d": [x1, y1, x2, y2], "points_2d": [[x1, y1], [x2, y2]], "jaw": s}, '
                   'with coordinates in [0, 1000].')

    def encode(self, g, gts):
        d = json.loads(super().encode(g, gts))
        return json.dumps({"bbox_2d": [_q(v) for v in bbox(gts)], **d})

    def _skip(self):
        return 4


FORMATS = {f.name: f for f in (XYWHT(), Endpoints(), EndpointsBox())}
