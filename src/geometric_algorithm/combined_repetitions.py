
import pandas as pd

SYNC_EXERCISES = {'09', '10', '11', '13', '14', '15'}
MIN_OVERLAP = 0.35

def overlap(a, b):
    start = max(a['start_s'], b['start_s'])
    end = min(a['end_s'], b['end_s'])
    duration = min(a['end_s'] - a['start_s'], b['end_s'] - b['start_s'])

    return max(0, end - start) / duration if duration > 0 else 0


def combine_repetitions(reps, chosen, subject, exercise, camera):
    exercise = str(exercise).zfill(2)
    rows = []
    sides = {side for side, limb in chosen}

    if not sides:
        return rows
    if not reps:
        return rows
    if len(sides) == 1 or exercise not in SYNC_EXERCISES:
        ordered = sorted(reps, key=lambda r: r['start_s'])

        for i, rep in enumerate(ordered, 1):
            rows.append({'subject': subject, 'exercise': exercise, 'camera': camera, 'repetition': i,
                'start_s': rep['start_s'], 'end_s': rep['end_s'], 'left_detected': rep['side'] == 'left',
                'right_detected': rep['side'] == 'right', 'confidence': 'single_limb', 'amplitude_deg': rep.get('amplitude_deg')})
        return rows

    left = sorted([r for r in reps if r['side'] == 'left'], key=lambda r: r['start_s'])
    right = sorted([r for r in reps if r['side'] == 'right'], key=lambda r: r['start_s'])

    matched_left = set()
    matched_right = set()
    combined = []
    candidates = []

    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score = overlap(a, b)

            if score >= MIN_OVERLAP:
                candidates.append((score, i, j))

    for score, i, j in sorted(candidates, reverse=True):
        if i in matched_left or j in matched_right:
            continue

        a, b = left[i], right[j]
        matched_left.add(i)
        matched_right.add(j)

        combined.append({'start_s': min(a['start_s'], b['start_s']), 'end_s': max(a['end_s'], b['end_s']),
            'left_detected': True, 'right_detected': True, 'confidence': 'both_limbs', 'amplitude_deg': None})

    for side, items, matched in (('left', left, matched_left), ('right', right, matched_right)):
        for i, rep in enumerate(items):
            if i in matched:
                continue

            combined.append({'start_s': rep['start_s'], 'end_s': rep['end_s'], 'left_detected': side == 'left',
                'right_detected': side == 'right', 'confidence': 'one_limb_only', 'amplitude_deg': rep.get('amplitude_deg')})

    for i, rep in enumerate(sorted(combined, key=lambda r: r['start_s']), 1):
        rows.append({'subject': subject, 'exercise': exercise, 'camera': camera, 'repetition': i, **rep})
    return rows
