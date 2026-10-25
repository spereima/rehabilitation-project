
import numpy as np
from tracking import coords
from collections import Counter
from config import REQUIRE_CONSENSUS_EVIDENCE
from config import (SKIP_S, WINDOW_S, MIN_COVERAGE, MIN_MOTION, BASELINE_S, TORSO_LIMIT)


def motion_scores(df):
    win = df[(df.time_s >= SKIP_S) & (df.time_s < SKIP_S + WINDOW_S)]
    scores = {}

    for side in ('left', 'right'):
        part = win[win.side == side]
        parts = {'leg': ('hip', 'knee', 'ankle', 'knee_angle'), 'arm': ('shoulder', 'elbow', 'wrist', 'elbow_angle')}

        for limb, (root, mid, tip, angle) in parts.items():
            good = part[[f'{j}_{a}' for j in (root, mid, tip) for a in 'xy']].notna().all(axis=1)
            good &= part[angle + '_status'].isin(['observed', 'estimated'])
            coverage = float(good.mean()) if len(good) else 0

            if good.sum() < 12 or coverage < MIN_COVERAGE:
                continue

            p = part.loc[good]
            length = np.median(np.linalg.norm(coords(p, mid) - coords(p, root), axis=1))
            if not np.isfinite(length) or length < 5:
                continue

            movement = (coords(p, tip) - coords(p, root)) / length
            spread = float(np.linalg.norm(np.quantile(movement, .9, axis=0) - np.quantile(movement, .1, axis=0)))
            ang = p[angle].dropna()
            angular = float((ang.quantile(.9) - ang.quantile(.1)) / 90) if len(ang) >= 12 else 0
            scores[side, limb] = {'M': spread, 'A': angular, 'score': .8 * spread + .2 * angular, 'coverage': coverage}

    return scores


def local_choice(scores):
    valid = {key: val['score'] for key, val in scores.items() if val['score'] >= MIN_MOTION}
    if not valid:
        return set()

    limb = max(valid, key=valid.get)[1]
    peak = max(score for (side, part), score in valid.items() if part == limb)
    return {key for key, score in valid.items() if key[1] == limb and score >= .35 * peak}


def vote(items):
    votes = Counter(key for item in items for key in item['local'])
    if not votes:
        return set()

    best = max(votes.values())
    return {key for key, count in votes.items() if count >= max(2, best - 1)}


def vote_rescue(items):
    common = vote(items)
    if not common:
        return common

    types = {limb for _, limb in common}
    votes = Counter(key for item in items for key in item['local'])

    for key, count in votes.items():
        if key in common or key[1] not in types or count < 2:
            continue

        support = sum(key in item['local'] and item['scores'].get(key, {}).get('coverage', 0) >= MIN_COVERAGE
            and item['scores'].get(key, {}).get('score', 0) >= MIN_MOTION for item in items)

        if support >= 2:
            common.add(key)

    return common


def annotate(df, item, common):
    df = df.copy()
    local, scores = item['local'], item['scores']

    if common:
        chosen = set(common)
        if REQUIRE_CONSENSUS_EVIDENCE:
            chosen &= set(scores)
    else:
        chosen = set(local)

    sources = {}
    for key in chosen:
        if key in local and key in common:
            sources[key] = 'local+consensus'
        elif key in common:
            sources[key] = 'consensus'
        else:
            sources[key] = 'local'

    show = lambda keys: ','.join(f'{side}_{limb}' for side, limb in sorted(keys)) or 'unknown'
    df['local_choice'] = show(local)
    df['consensus_choice'] = show(common)
    df['selected'] = show(chosen)
    df['active_limb'] = next(iter(chosen))[1] if chosen else None
    df['active_sides'] = ','.join(side for side in ('left', 'right') if (side, df['active_limb'].iloc[0]) in chosen) if chosen else ''
    df['is_active'] = [(side, limb) in chosen for side, limb in zip(df.side, df.active_limb)]
    df['decision_source'] = [sources.get((side, limb), 'unknown') for side, limb in zip(df.side, df.active_limb)]

    for limb in ('arm', 'leg'):
        for side in ('left', 'right'):
            key = side, limb
            df[f'{side}_{limb}_active'] = key in chosen
            df[f'{side}_{limb}_score'] = scores.get(key, {}).get('score', np.nan)

    df['movement_score'] = [scores.get((side, limb), {}).get('score', np.nan) for side, limb in zip(df.side, df.active_limb)]
    df['movement_coverage'] = [scores.get((side, limb), {}).get('coverage', np.nan) for side, limb in zip(df.side, df.active_limb)]

    ref = df[(df.time_s >= BASELINE_S[0]) & (df.time_s < BASELINE_S[1])].drop_duplicates('frame').torso_tilt_signed.dropna()
    baseline = float(ref.median()) if len(ref) >= 5 else np.nan
    df['torso_reference'] = baseline
    df['torso_change'] = (df.torso_tilt_signed - baseline).abs()
    df['torso_status'] = np.select([df.torso_change.isna(), df.torso_change <= TORSO_LIMIT], ['UNKNOWN', 'STABLE'], default='CHANGED')
    return df, chosen


def with_weight(scores, weight):
    return {key: {**val, 'score': weight * val['M'] + (1 - weight) * val['A']} for key, val in scores.items()}
