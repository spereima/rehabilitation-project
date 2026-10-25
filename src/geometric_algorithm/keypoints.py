import cv2
import numpy as np
from scipy.spatial import cKDTree
from config import MIN_AREA, SEGMENT_OF
from segmentation import mask_components, axis_ends

def opposite_endpoint(mask, near, offset=(0, 0)):
    pair = axis_ends(mask, offset)
    return max(pair, key=lambda p: np.linalg.norm(p - near)) if pair is not None and near is not None else None


def contour_points(mask, offset=(0, 0)):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)
    return np.concatenate(contours).reshape(-1, 2).astype(float)


def junction(a, b, limit, usable, outline):
    if usable[a] < MIN_AREA or usable[b] < MIN_AREA:
        return None, np.nan

    pa, pb = outline(a), outline(b)
    d, idx = cKDTree(pb).query(pa)
    best = float(d.min())
    if best > limit:
        return None, best

    nearest = d <= min(limit, best + 3.0)
    mid = (pa[nearest] + pb[idx[nearest]]) / 2
    return np.median(mid, axis=0), best


def pack_point(out, name, p, reason='observed'):
    out[name + '_x'] = float(p[0]) if p is not None else np.nan
    out[name + '_y'] = float(p[1]) if p is not None else np.nan
    out[name + '_status'] = reason if p is not None or reason != 'observed' else 'missing'


def extract_frame(frame):
    masks, count, offset = mask_components(frame)
    torso = axis_ends(masks['torso'], offset)
    top, bottom = sorted(torso, key=lambda p: p[1]) if torso else (None, None)
    scale = float(np.linalg.norm(top - bottom)) if torso else frame.shape[0] * .18
    scale = max(scale, 30.)
    limit = float(np.clip(.25 * scale, 10, 65))
    usable = {name: int(cv2.countNonZero(mask)) for name, mask in masks.items()}
    contours, result = {}, []

    def outline(name):
        if name not in contours:
            contours[name] = contour_points(masks[name], offset)
        return contours[name]

    for side in ('left', 'right'):
        m = {part: masks[f'{side}_{part}'] for part in ('thigh', 'shin', 'foot', 'upper_arm', 'lower_arm', 'hand')}
        s = lambda part: f'{side}_{part}'
        knee, kd = junction(s('thigh'), s('shin'), limit, usable, outline)
        ankle, ad = junction(s('shin'), s('foot'), limit, usable, outline)
        elbow, ed = junction(s('upper_arm'), s('lower_arm'), limit, usable, outline)
        wrist, wd = junction(s('lower_arm'), s('hand'), limit, usable, outline)

        positions = {'hip': opposite_endpoint(m['thigh'], knee, offset) if knee is not None else None,
            'knee': knee, 'ankle': ankle, 'toe': opposite_endpoint(m['foot'], ankle, offset) if ankle is not None else None,
            'shoulder': opposite_endpoint(m['upper_arm'], elbow, offset) if elbow is not None else None, 'elbow': elbow,
            'wrist': wrist, 'hand': opposite_endpoint(m['hand'], wrist, offset) if wrist is not None else None,
            'torso_top': top, 'torso_bottom': bottom}
        out = {'side': side, 'body_scale': scale}

        for part in m:
            out[f'{part}_pixels'] = count[f'{side}_{part}']
            out[f'{part}_usable_pixels'] = usable[s(part)]

        out['torso_pixels'] = count['torso']

        for name, dist in (('knee', kd), ('ankle', ad), ('elbow', ed), ('wrist', wd)):
            out[name + '_junction_gap_px'] = dist

        for name, pos in positions.items():
            missing = [p for p in SEGMENT_OF[name].split('+') if (count['torso'] if p == 'torso' else count[f'{side}_{p}']) < MIN_AREA]
            why = 'mask_missing:' + ','.join(missing) if missing else 'junction_or_axis_unavailable' if pos is None else 'observed'
            pack_point(out, name, pos, why)

        result.append(out)

    return result
