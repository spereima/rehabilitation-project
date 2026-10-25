
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from smoothness import smoothness
from config import REPETITION_START_S


def count_repetitions(df, chosen, subj, exercise, camera):
    rows = []
    ts = df.drop_duplicates('frame').sort_values('frame').time_s.to_numpy(float)
    dt = np.median(np.diff(ts)) if len(ts) > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return rows

    fps = 1 / dt
    for side, limb in sorted(chosen):
        data = df[(df.side == side) & (df.time_s >= REPETITION_START_S)].sort_values('frame')
        if len(data) < max(20, round(2 * fps)):
            continue

        candidates = ('knee_angle', 'hip_angle_2d', 'ankle_angle') if limb == 'leg' else ('elbow_angle', 'shoulder_angle_2d', 'wrist_angle')
        best = None

        for name in candidates:
            column = name + '_smooth'
            if column not in data:
                continue

            raw = data[column].to_numpy(float)
            coverage = np.isfinite(raw).mean()
            if coverage < .45:
                continue

            valid = raw[np.isfinite(raw)]
            span = np.percentile(valid, 90) - np.percentile(valid, 10)
            quality = span * coverage
            if best is None or quality > best[0]:
                best = (quality, name, raw)

        if best is None or best[0] < 8:
            continue

        _, angle_name, raw = best
        x = pd.Series(raw).interpolate(limit=max(2, round(.25 * fps)), limit_direction='both', limit_area='inside')
        present = x.notna().to_numpy()
        boundaries = np.r_[0, np.where(np.diff(present.astype(int)) != 0)[0] + 1, len(x)]
        count = 0
        times = data.time_s.to_numpy(float)

        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if not present[start] or stop - start < round(2 * fps):
                continue

            y = x.iloc[start:stop].to_numpy(float)
            window = max(5, round(.55 * fps) | 1)
            window = min(window, len(y) if len(y) % 2 else len(y) - 1)
            if window >= 5:
                y = savgol_filter(y, window, 2)

            span = np.percentile(y, 90) - np.percentile(y, 10)
            prominence = max(6., .30 * span)
            if span < 9:
                continue

            gap = max(3, round(1.7 * fps))
            high, _ = find_peaks(y, prominence=prominence, distance=gap)
            low, _ = find_peaks(-y, prominence=prominence, distance=gap)

            for edge in (0, len(y) - 1):
                if y[edge] <= np.percentile(y, 15) + 4:
                    low = np.unique(np.r_[low, edge])
                if y[edge] >= np.percentile(y, 85) - 4:
                    high = np.unique(np.r_[high, edge])

            options = []
            for troughs, peaks, direction in ((low, high, 'extension'), (high, low, 'flexion')):
                cycles = []

                for a, b in zip(troughs[:-1], troughs[1:]):
                    middle = peaks[(peaks > a) & (peaks < b)]
                    if not len(middle):
                        continue

                    peak = middle[np.argmax(np.abs(y[middle] - y[a]))]
                    amplitude = float(max(y[a:b + 1]) - min(y[a:b + 1]))
                    if amplitude < max(10., .35 * span) or abs(y[a] - y[b]) > max(12., .4 * amplitude) or (b - a) / fps < 2.:
                        continue

                    cycles.append((a, b, amplitude, direction))

                options.append(cycles)

            selected = max(options, key=lambda cycles: (len(cycles), sum(c[2] for c in cycles)))

            for a, b, amplitude, direction in selected:
                count += 1
                rows.append({
                    'subject': subj, 'exercise': exercise, 'camera': camera, 'side': side, 'limb': limb, 'repetition': count,
                    'angle_source': angle_name, 'start_s': round(float(times[start + a]), 3), 'end_s': round(float(times[start + b]), 3),
                    'amplitude_deg': round(amplitude, 2), 'smoothness_rms_accel': round(float(smoothness(y[a:b + 1], fps)), 2),
                    'movement_type': direction
                })

    return rows


def overlap(a, b):
    start, end = max(a['start_s'], b['start_s']), min(a['end_s'], b['end_s'])
    duration = min(a['end_s'] - a['start_s'], b['end_s'] - b['start_s'])
    return max(0, end - start) / duration if duration > 0 else 0


def count_exercise_repetitions(reps, chosen, subj, exercise, camera):
    rows = []

    def add(a, b=None):
        both = b is not None
        rows.append({
            'subject': subj, 'exercise': exercise, 'camera': camera, 'repetition': 0,
            'start_s': round(min(a['start_s'], b['start_s']) if both else a['start_s'], 3),
            'end_s': round(max(a['end_s'], b['end_s']) if both else a['end_s'], 3),
            'left_detected': both or a['side'] == 'left', 'right_detected': both or a['side'] == 'right',
            'confidence': 'both_limbs' if both else 'one_limb_only'})
    if not reps:
        return rows

    reps = [r for r in reps if (r['side'], r['limb']) in chosen]
    left = sorted([r for r in reps if r['side'] == 'left'], key=lambda r: r['start_s'])
    right = sorted([r for r in reps if r['side'] == 'right'], key=lambda r: r['start_s'])
    candidates = []

    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a['limb'] != b['limb']:
                continue

            score = overlap(a, b)
            if score >= .35:
                candidates.append((score, i, j))

    used_left, used_right = set(), set()
    for score, i, j in sorted(candidates, reverse=True):
        if i in used_left or j in used_right:
            continue

        used_left.add(i)
        used_right.add(j)
        add(left[i], right[j])

    for side_reps, used in ((left, used_left), (right, used_right)):
        for i, rep in enumerate(side_reps):
            if i in used:
                continue

            matched_duplicate = any(row['confidence'] == 'both_limbs' and overlap(rep, row) >= .65 for row in rows)
            if not matched_duplicate:
                add(rep)

    rows.sort(key=lambda r: r['start_s'])
    for i, row in enumerate(rows, 1):
        row['repetition'] = i

    return rows
