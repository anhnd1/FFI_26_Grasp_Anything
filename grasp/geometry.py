"""Grasp rectangle geometry.

A grasp is (x, y, w, h, theta): center in pixels, w = jaw opening along theta, h = jaw size across it,
theta in degrees. Grasps are 180-degree symmetric, so theta is kept in [0, 180).
Image coordinates: x right, y down (as in Grasp-Anything++).
"""
import math

IMG = 416  # Grasp-Anything++ image size
NORM = 1000  # Qwen grounding frame: relative coordinates in [0, 1000]


def norm_theta(t):
    return t % 180.0


def angle_diff(a, b):
    """Smallest difference between two grasp angles (degrees), respecting 180-degree symmetry."""
    d = abs(norm_theta(a) - norm_theta(b))
    return min(d, 180.0 - d)


def corners(x, y, w, h, theta):
    a = math.radians(theta)
    dx, dy = math.cos(a), math.sin(a)
    return [(x + sw * w / 2 * dx - sh * h / 2 * dy, y + sw * w / 2 * dy + sh * h / 2 * dx)
            for sw, sh in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def to_endpoints(x, y, w, h, theta):
    """Rectangle -> (p1, p2, h): jaw endpoints in canonical order (smaller x first, ties by smaller y)."""
    a = math.radians(theta)
    ox, oy = w / 2 * math.cos(a), w / 2 * math.sin(a)
    p1, p2 = (x - ox, y - oy), (x + ox, y + oy)
    if (p2[0], p2[1]) < (p1[0], p1[1]):
        p1, p2 = p2, p1
    return p1, p2, h


def from_endpoints(p1, p2, h):
    """Exact inverse of to_endpoints."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, math.hypot(dx, dy), h,
            norm_theta(math.degrees(math.atan2(dy, dx))))


def endpoint_dist(g1, g2):
    """Distance between two grasps in endpoint space, invariant to endpoint order (pixels)."""
    a1, a2, ah = to_endpoints(*g1)
    b1, b2, bh = to_endpoints(*g2)
    d = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])
    return min(d(a1, b1) + d(a2, b2), d(a1, b2) + d(a2, b1)) + abs(ah - bh)


def medoid_index(grasps):
    """Index of the grasp with minimal total endpoint distance to all others (consensus grasp)."""
    return min(range(len(grasps)), key=lambda i: sum(endpoint_dist(grasps[i], g) for g in grasps))


def bbox(grasps):
    """Axis-aligned box (x1, y1, x2, y2) covering all grasp rectangles, clipped to the image."""
    pts = [p for g in grasps for p in corners(*g)]
    clip = lambda v: min(max(v, 0.0), IMG)
    return (clip(min(p[0] for p in pts)), clip(min(p[1] for p in pts)),
            clip(max(p[0] for p in pts)), clip(max(p[1] for p in pts)))


def to_norm(v):
    return v / IMG * NORM


def from_norm(v):
    return v / NORM * IMG


# ---- rotated-rectangle IoU (convex polygon clipping, no shapely dependency) ----

def _area(poly):
    return abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) / 2


def _clip(subject, clipper):
    """Sutherland-Hodgman clipping of a convex/any polygon by a convex polygon."""
    out = subject
    # orientation of the clipper decides which side is "inside"
    sign = 1 if sum(clipper[i][0] * clipper[i - 1][1] - clipper[i - 1][0] * clipper[i][1]
                    for i in range(len(clipper))) < 0 else -1
    for i in range(len(clipper)):
        a, b = clipper[i - 1], clipper[i]
        side = lambda p: sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            sp, sq = side(p), side(q)
            if sq >= 0:
                if sp < 0:
                    out.append(_intersect(p, q, sp, sq))
                out.append(q)
            elif sp >= 0:
                out.append(_intersect(p, q, sp, sq))
        if not out:
            return []
    return out


def _intersect(p, q, sp, sq):
    t = sp / (sp - sq)
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def iou(g1, g2):
    c1, c2 = corners(*g1), corners(*g2)
    a1, a2 = _area(c1), _area(c2)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    inter = _clip(c1, c2)
    i = _area(inter) if len(inter) >= 3 else 0.0
    return i / (a1 + a2 - i)


def is_success(pred, gts, iou_thr=0.25, angle_thr=30.0):
    """Standard rectangle metric: matches ANY ground-truth grasp with IoU > thr and angle error < thr."""
    return any(angle_diff(pred[4], g[4]) < angle_thr and iou(pred, g) > iou_thr for g in gts)
