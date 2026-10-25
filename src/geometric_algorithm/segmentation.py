
import cv2
import numpy as np
from config import COLORS, COLOR_TOLERANCE, MIN_AREA
NAMES = list(COLORS)

def colour_tables():
    bgr = [tuple(bytes.fromhex(code[1:]))[::-1] for code in COLORS.values()]
    levels = sorted({v for colour in bgr for v in colour})
    none, base = len(levels), len(levels) + 1

    if base ** 3 > 256:
        raise ValueError('Too many colour levels for the lookup table')

    level = np.full(256, none, np.uint8)
    for i, v in enumerate(levels):
        lo, hi = max(v - COLOR_TOLERANCE, 0), min(v + COLOR_TOLERANCE, 255)
        if (level[lo:hi + 1] != none).any():
            raise ValueError('COLOR_TOLERANCE is too large: colour ranges overlap')
        level[lo:hi + 1] = i

    channels = np.dstack([level * base * base, level * base, level]).reshape(1, 256, 3).astype(np.uint8)
    part = np.zeros(256, np.uint8)

    for i, (b, g, r) in enumerate(bgr, 1):
        code = int(level[b]) * base * base + int(level[g]) * base + int(level[r])
        part[code] = i

    return channels, part


CHANNELS, PART = colour_tables()
SUM = np.ones((1, 3), np.float32)


def label_image(frame):
    code = cv2.transform(cv2.LUT(frame, CHANNELS), SUM)
    return cv2.LUT(code, PART)


def crop_box(mask, pad=2):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0:
        return 0, 0, min(2, mask.shape[1]), min(2, mask.shape[0])
    return (max(x - pad, 0) & ~1, max(y - pad, 0) & ~1,
        min(x + w + pad, mask.shape[1]), min(y + h + pad, mask.shape[0]))


def mask_components(frame):
    labels_all = label_image(frame)
    x0, y0, x1, y1 = crop_box(labels_all)
    crop = labels_all[y0:y1, x0:x1]
    empty = np.zeros(crop.shape, np.uint8)
    result, counts = {}, {}

    for i, name in enumerate(NAMES, 1):
        raw = cv2.compare(crop, float(i), cv2.CMP_EQ)
        counts[name] = int(cv2.countNonZero(raw))
        result[name] = empty
        if counts[name] < MIN_AREA:
            continue

        bx0, by0, bx1, by1 = crop_box(raw)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw[by0:by1, bx0:bx1], 8)
        candidates = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= MIN_AREA]

        if candidates:
            k = max(candidates, key=lambda j: stats[j, cv2.CC_STAT_AREA])
            mask = empty.copy()
            mask[by0:by1, bx0:bx1] = cv2.compare(labels, float(k), cv2.CMP_EQ)
            result[name] = mask

    return result, counts, (x0, y0)


def pixels(mask, offset=(0, 0)):
    yy, xx = np.nonzero(mask)
    return np.column_stack([xx + offset[0], yy + offset[1]]).astype(float)


def centroid(mask, offset=(0, 0)):
    p = pixels(mask, offset)
    return p.mean(axis=0) if len(p) >= MIN_AREA else None


def axis_ends(mask, offset=(0, 0)):
    p = pixels(mask, offset)
    if len(p) < MIN_AREA:
        return None

    mu = p.mean(axis=0)
    _, eigenvectors = cv2.PCACompute(p.astype(np.float32), mean=np.array([mu], np.float32), maxComponents=1)
    axis = eigenvectors[0].astype(float)
    t = (p - mu) @ axis
    return mu + np.quantile(t, .05) * axis, mu + np.quantile(t, .95) * axis
