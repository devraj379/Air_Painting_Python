"""Offline computer-vision & geometric AI assistant for Air Painting Studio."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _to_array(points: list[dict[str, float]]) -> np.ndarray:
    """Convert [{x, y}, ...] points into a deduplicated Nx2 float32 array."""
    cleaned: list[list[float]] = []
    for pt in points:
        x = float(pt["x"])
        y = float(pt["y"])
        if not cleaned or math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) >= 1.5:
            cleaned.append([x, y])
    return np.array(cleaned, dtype=np.float32)


def _to_dicts(arr: np.ndarray) -> list[dict[str, float]]:
    """Convert an Nx2 array back into [{x, y}, ...] JSON-serializable points."""
    return [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)} for p in arr]


def _smooth_open_stroke(pts: np.ndarray) -> np.ndarray:
    """Apply light Douglas-Peucker + Chaikin corner-cutting for smooth freehand curves."""
    if len(pts) < 4:
        return pts
    contour = pts.reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon=2.2, closed=False).reshape(-1, 2)
    if len(simplified) < 3:
        return pts
    # Two passes of Chaikin smoothing while preserving start and end points
    curr = simplified
    for _ in range(2):
        new_pts = [curr[0]]
        for i in range(len(curr) - 1):
            p0, p1 = curr[i], curr[i + 1]
            new_pts.append(0.75 * p0 + 0.25 * p1)
            new_pts.append(0.25 * p0 + 0.75 * p1)
        new_pts.append(curr[-1])
        curr = np.array(new_pts, dtype=np.float32)
    return curr


def _ideal_star_contour(cx: float, cy: float, outer_r: float, inner_r: float) -> np.ndarray:
    """Generate a canonical 5-point star contour centered at (cx, cy)."""
    verts = []
    for i in range(11):
        angle = -math.pi / 2 + i * (math.pi / 5)
        r = outer_r if i % 2 == 0 else inner_r
        verts.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
    return np.array(verts, dtype=np.float32)


def _ideal_heart_contour(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Generate a canonical heart contour scaled to (w, h) at (cx, cy)."""
    ts = np.linspace(0, 2 * math.pi, 65)
    xs = 16 * (np.sin(ts) ** 3)
    ys = -(13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts))
    xs = (xs - xs.mean()) / (xs.max() - xs.min() + 1e-6) * w + cx
    ys = (ys - ys.mean()) / (ys.max() - ys.min() + 1e-6) * h + cy
    return np.column_stack([xs, ys]).astype(np.float32)


def _resample_closed(pts: np.ndarray, n: int = 64) -> np.ndarray:
    """Resample a closed polyline into n evenly spaced points."""
    closed = np.vstack([pts, pts[0]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        return pts
    targets = np.linspace(0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


def _count_star_peaks(pts: np.ndarray, cx: float, cy: float) -> int:
    """Count distinct radial peaks around centroid for star detection."""
    sampled = _resample_closed(pts, 60)
    radii = np.hypot(sampled[:, 0] - cx, sampled[:, 1] - cy)
    r_mean = float(np.mean(radii))
    if r_mean < 1e-3:
        return 0
    norm = radii / r_mean
    # Circular smoothing
    padded = np.concatenate([norm[-2:], norm, norm[:2]])
    smooth = np.convolve(padded, np.ones(5) / 5.0, mode="valid")
    peaks = []
    n = len(smooth)
    for i in range(n):
        prev_v = smooth[(i - 1) % n]
        next_v = smooth[(i + 1) % n]
        if smooth[i] >= prev_v and smooth[i] > next_v and smooth[i] > 1.12:
            if not peaks or (i - peaks[-1]) >= 6:
                peaks.append(i)
    if len(peaks) >= 2 and (n - peaks[-1] + peaks[0]) < 6:
        peaks.pop()
    return len(peaks)


def _has_heart_top_cleft(pts: np.ndarray, cx: float, y_min: float, w: float, h: float) -> bool:
    """Verify the upper center dips below the left and right upper lobes."""
    upper = pts[pts[:, 1] <= y_min + 0.55 * h]
    left_top = upper[upper[:, 0] < cx - 0.12 * w]
    right_top = upper[upper[:, 0] > cx + 0.12 * w]
    mid_band = upper[np.abs(upper[:, 0] - cx) <= 0.08 * w]
    if len(left_top) == 0 or len(right_top) == 0 or len(mid_band) == 0:
        return False
    left_peak_y = float(np.min(left_top[:, 1]))
    right_peak_y = float(np.min(right_top[:, 1]))
    mid_top_y = float(np.min(mid_band[:, 1]))
    lobe_y = (left_peak_y + right_peak_y) / 2.0
    return (mid_top_y - lobe_y) >= 0.06 * h and abs(left_peak_y - right_peak_y) <= 0.25 * h


def beautify_stroke(points: list[dict[str, float]]) -> dict[str, Any]:
    """Classify a single stroke and return clean beautified geometry."""
    pts = _to_array(points)
    if len(pts) < 2:
        return {
            "shape": "Dot",
            "confidence": 0.9,
            "beautified_points": _to_dicts(pts),
        }

    diffs = np.diff(pts, axis=0)
    arc_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    if arc_len < 12:
        return {
            "shape": "Dot",
            "confidence": 0.9,
            "beautified_points": _to_dicts(pts),
        }

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    w = float(x_max - x_min)
    h = float(y_max - y_min)
    diag = math.hypot(w, h)
    chord = float(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))

    # 1. Straight Line detection
    if chord >= 20 and arc_len <= 1.18 * chord:
        v = pts[-1] - pts[0]
        v_norm = math.hypot(float(v[0]), float(v[1]))
        if v_norm > 1e-6:
            perp = np.abs(np.cross(pts - pts[0], v)) / v_norm
            max_perp = float(np.max(perp))
            if max_perp <= max(9.0, 0.08 * chord):
                conf = round(max(0.75, min(0.98, 1.0 - max_perp / (0.12 * chord + 1e-5))), 2)
                return {
                    "shape": "Line",
                    "confidence": conf,
                    "beautified_points": _to_dicts(np.array([pts[0], pts[-1]], dtype=np.float32)),
                }

    # 2. Closed-shape detection
    is_closed = chord <= max(32.0, 0.36 * diag)
    if is_closed and w >= 16 and h >= 16:
        closed_pts = np.vstack([pts, pts[0]])
        cnt = closed_pts.reshape(-1, 1, 2).astype(np.float32)
        area = abs(float(cv2.contourArea(cnt)))
        perim = float(cv2.arcLength(cnt, True))

        if area >= 150 and perim > 10:
            moments = cv2.moments(cnt)
            if moments["m00"] > 1e-6:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx, cy = float((x_min + x_max) / 2), float((y_min + y_max) / 2)

            aspect = w / (h + 1e-6)
            circularity = (4.0 * math.pi * area) / (perim * perim)
            radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            mean_r = float(np.mean(radii))
            cv_r = float(np.std(radii) / (mean_r + 1e-6))

            # 2a. 5-Point Star detection
            peaks = _count_star_peaks(pts, cx, cy)
            if peaks == 5:
                outer_r = float(np.percentile(radii, 92))
                inner_r = float(np.percentile(radii, 20))
                if outer_r > 1.35 * max(inner_r, 1.0):
                    star_pts = _ideal_star_contour(cx, cy, outer_r, max(inner_r, outer_r * 0.42))
                    return {
                        "shape": "Star",
                        "confidence": 0.9,
                        "beautified_points": _to_dicts(star_pts),
                    }

            # 2b. Heart detection
            if 0.65 <= aspect <= 1.45 and _has_heart_top_cleft(pts, cx, float(y_min), w, h):
                ideal_heart = _ideal_heart_contour(float((x_min + x_max) / 2), float((y_min + y_max) / 2), w, h)
                dist = cv2.matchShapes(cnt, ideal_heart.reshape(-1, 1, 2), cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < 0.35:
                    conf = round(max(0.75, min(0.95, 1.0 - dist)), 2)
                    return {
                        "shape": "Heart",
                        "confidence": conf,
                        "beautified_points": _to_dicts(ideal_heart),
                    }

            # 2c. Polygon approximation (Triangle / Square / Rectangle / Diamond)
            approx = cv2.approxPolyDP(cnt, epsilon=0.045 * perim, closed=True).reshape(-1, 2)
            if len(approx) == 3:
                tri_pts = np.vstack([approx, approx[0]])
                return {
                    "shape": "Triangle",
                    "confidence": 0.89,
                    "beautified_points": _to_dicts(tri_pts),
                }
            if len(approx) == 4:
                bbox_fill = area / (w * h + 1e-6)
                if bbox_fill >= 0.72:
                    rect_pts = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                        [x_min, y_min],
                    ], dtype=np.float32)
                    shape_name = "Square" if 0.82 <= aspect <= 1.22 else "Rectangle"
                    conf = round(min(0.95, 0.65 + 0.3 * bbox_fill), 2)
                    return {
                        "shape": shape_name,
                        "confidence": conf,
                        "beautified_points": _to_dicts(rect_pts),
                    }
                elif 0.40 <= bbox_fill < 0.72:
                    diamond_pts = np.array([
                        [(x_min + x_max) / 2, y_min],
                        [x_max, (y_min + y_max) / 2],
                        [(x_min + x_max) / 2, y_max],
                        [x_min, (y_min + y_max) / 2],
                        [(x_min + x_max) / 2, y_min],
                    ], dtype=np.float32)
                    return {
                        "shape": "Diamond",
                        "confidence": 0.82,
                        "beautified_points": _to_dicts(diamond_pts),
                    }

            # 2d. Circle / Oval detection
            if circularity >= 0.78 and cv_r <= 0.17 and len(approx) >= 5:
                angles = np.linspace(0, 2 * math.pi, 49)
                if 0.80 <= aspect <= 1.25:
                    circle_pts = np.column_stack([
                        cx + mean_r * np.cos(angles),
                        cy + mean_r * np.sin(angles),
                    ])
                    conf = round(max(0.8, min(0.97, 0.98 - cv_r)), 2)
                    return {
                        "shape": "Circle",
                        "confidence": conf,
                        "beautified_points": _to_dicts(circle_pts),
                    }
                else:
                    oval_pts = np.column_stack([
                        (x_min + x_max) / 2 + (w / 2) * np.cos(angles),
                        (y_min + y_max) / 2 + (h / 2) * np.sin(angles),
                    ])
                    return {
                        "shape": "Oval",
                        "confidence": 0.84,
                        "beautified_points": _to_dicts(oval_pts),
                    }

    # 3. Fallback: smooth freehand curve
    smoothed = _smooth_open_stroke(pts)
    return {
        "shape": "Freehand",
        "confidence": 0.6,
        "beautified_points": _to_dicts(smoothed),
    }


def _bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = np.min(pts, axis=0)
    x1, y1 = np.max(pts, axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float], margin: float = 12.0) -> bool:
    return (
        inner[0] >= outer[0] - margin
        and inner[2] <= outer[2] + margin
        and inner[1] >= outer[1] - margin
        and inner[3] <= outer[3] + margin
    )


def analyze_canvas(strokes: list[dict[str, Any]], target_challenge: str | None = None) -> dict[str, Any]:
    """Recognize individual shapes and composite doodles across all canvas strokes."""
    active_strokes = [s for s in strokes if not s.get("isEraser") and len(s.get("points", [])) >= 1]
    if not active_strokes:
        return {
            "guess": "Blank Canvas",
            "confidence": 100,
            "details": [],
            "tip": "Draw a Circle, Triangle, Square, Star, Heart, House, or Smiley Face in the air!",
            "challenge_match": False,
        }

    classified: list[dict[str, Any]] = []
    for s in active_strokes:
        pts = _to_array(s["points"])
        res = beautify_stroke(s["points"])
        box = _bbox(pts)
        classified.append({
            "shape": res["shape"],
            "confidence": res["confidence"],
            "box": box,
            "w": box[2] - box[0],
            "h": box[3] - box[1],
            "cx": (box[0] + box[2]) / 2.0,
            "cy": (box[1] + box[3]) / 2.0,
        })

    counts: dict[str, int] = {}
    for item in classified:
        counts[item["shape"]] = counts.get(item["shape"], 0) + 1

    details = [f"{v} {k}" for k, v in counts.items()]

    # Composite Scene Recognition
    guess = None
    confidence = 0

    # 1. Smiley Face: Large Circle/Oval containing >= 2 small features in top half + 1 feature in bottom half
    for idx, item in enumerate(classified):
        if item["shape"] in ("Circle", "Oval") and item["w"] >= 60 and item["h"] >= 60:
            inside = [other for j, other in enumerate(classified) if j != idx and _contains(item["box"], other["box"])]
            if len(inside) >= 3:
                top_features = [o for o in inside if o["cy"] < item["cy"] + 0.05 * item["h"]]
                bot_features = [o for o in inside if o["cy"] >= item["cy"]]
                if len(top_features) >= 2 and len(bot_features) >= 1:
                    guess = "Smiley Face"
                    confidence = 89
                    break

    # 2. House: Triangle positioned directly above/overlapping a Square or Rectangle
    if not guess:
        triangles = [c for c in classified if c["shape"] == "Triangle"]
        rects = [c for c in classified if c["shape"] in ("Square", "Rectangle")]
        for tri in triangles:
            for rect in rects:
                horiz_aligned = abs(tri["cx"] - rect["cx"]) <= max(tri["w"], rect["w"]) * 0.45
                vert_stacked = tri["box"][3] <= rect["box"][1] + rect["h"] * 0.35 and tri["box"][1] < rect["box"][1]
                if horiz_aligned and vert_stacked:
                    guess = "House"
                    confidence = 88
                    break
            if guess:
                break

    # 3. Sun: Circle surrounded by >= 4 small Lines
    if not guess:
        circles = [c for c in classified if c["shape"] in ("Circle", "Oval") and c["w"] >= 35]
        lines = [c for c in classified if c["shape"] == "Line"]
        for circ in circles:
            nearby_rays = 0
            for ln in lines:
                dist = math.hypot(ln["cx"] - circ["cx"], ln["cy"] - circ["cy"])
                if circ["w"] * 0.35 <= dist <= circ["w"] * 1.4:
                    nearby_rays += 1
            if nearby_rays >= 4:
                guess = "Sun"
                confidence = 86
                break

    # 4. Single prominent shape fallback
    if not guess:
        non_free = [c for c in classified if c["shape"] not in ("Freehand", "Dot")]
        if non_free:
            best = max(non_free, key=lambda c: c["w"] + c["h"])
            guess = best["shape"]
            confidence = int(round(best["confidence"] * 100))
        else:
            guess = "Freehand Doodle"
            confidence = 68

    # Helpful tip
    tips = {
        "Circle": "Try adding two dots and a smile inside to turn it into a Smiley Face!",
        "Square": "Draw a Triangle right on top to build a House!",
        "Rectangle": "Draw a Triangle right on top to build a House!",
        "Triangle": "Add a Square underneath it to make a House!",
        "Smiley Face": "Great expression! Try Beautify to polish the outline.",
        "House": "Nice architecture! You can add a Sun in the corner.",
        "Star": "Sharp star! Try drawing a Heart or House next.",
        "Heart": "Clean heart! Try changing colors or adding a border.",
        "Sun": "Bright and sunny!",
    }
    tip = tips.get(guess, f"Detected {len(active_strokes)} stroke(s): {', '.join(details)}.")

    challenge_match = False
    if target_challenge:
        challenge_match = guess.lower() == target_challenge.lower()

    return {
        "guess": guess,
        "confidence": confidence,
        "details": details,
        "tip": tip,
        "challenge_match": challenge_match,
    }
