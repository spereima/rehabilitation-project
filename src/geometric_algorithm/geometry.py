import numpy as np
import pandas as pd
from config import (TRIPLES, CHAINS, POINTS, ANGLE_JUMP_DEG)
from tracking import coords

def geometry(df):
    df = df.copy()
    edges = {'thigh': ('hip', 'knee'), 'shin': ('knee', 'ankle'), 'upper_arm': ('shoulder', 'elbow'), 'lower_arm': ('elbow', 'wrist')}

    for name, (a, b) in edges.items():
        d = np.linalg.norm(coords(df, a) - coords(df, b), axis=1)
        observed = ((df[a + '_status'] == 'observed') & (df[b + '_status'] == 'observed')).to_numpy()
        reference_values = d[observed & np.isfinite(d)]
        reference = (float(np.median(reference_values)) if len(reference_values) >= 10 else np.nan)

        if not np.isfinite(reference):
            reference_values = d[np.isfinite(d)]
            reference = (float(np.median(reference_values)) if len(reference_values) >= 10 else np.nan)

        df[name + '_length_px'] = d
        df[name + '_reference_px'] = reference
        df[name + '_geometry_ok'] = (np.isfinite(reference) & (reference >= 5) & np.isfinite(d) & (d >= 0.48 * reference) & (d <= 1.65 * reference))
    return df


def angles_and_diagnostics(df, fps):
    df = geometry(df)

    for metric, triple in TRIPLES.items():
        a, b, c = (coords(df, j) for j in triple)
        u, v = a - b, c - b
        norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        z = np.divide(np.sum(u * v, axis=1), norm, out=np.full(len(df), np.nan), where=norm > 1e-06)
        values = np.degrees(np.arccos(np.clip(z, -1, 1)))
        needs = [('thigh', ('hip', 'knee')), ('shin', ('knee', 'ankle')), ('upper_arm', ('shoulder', 'elbow')), ('lower_arm', ('elbow', 'wrist'))]
        valid_edges = [seg for seg, joints in needs if set(joints).issubset(triple)]

        good_geom = np.ones(len(df), bool)

        for edge in valid_edges:
            good_geom &= df[edge + '_geometry_ok'].to_numpy(bool)

        absent = ~np.isfinite(np.column_stack([a, b, c])).all(axis=1)
        invalid = ~good_geom & ~absent
        values[invalid] = np.nan

        reason = np.full(len(df), 'observed', dtype=object)
        reason[absent] = 'point_missing'
        reason[invalid] = 'geometry_rejected'
        point_status = df[[j + '_status' for j in triple]]
        estimated = (point_status.eq('estimated').any(axis=1).to_numpy() & ~absent & ~invalid)
        reason[estimated] = 'estimated'

        smooth = pd.Series(values).rolling(5, center=True, min_periods=1).median().to_numpy()
        smooth[~np.isfinite(values)] = np.nan
        delta = np.abs(np.diff(smooth, prepend=np.nan))
        jump = (np.isfinite(delta) &(delta > ANGLE_JUMP_DEG))

        df[metric] = values
        df[metric + '_smooth'] = smooth
        df[metric + '_status'] = reason
        df[metric + '_jump'] = jump
        df[metric + '_speed_deg_s'] = delta * fps
        df[metric + '_confidence'] = (np.min(np.column_stack([df[j + '_confidence'].to_numpy() for j in triple]), axis=1) * good_geom)

    for limb in CHAINS:
        main = (('knee_angle', 'hip_angle_2d') if limb == 'leg' else ('elbow_angle', 'shoulder_angle_2d'))
        df[limb + '_valid'] = df[[name + '_smooth' for name in main]].notna().all(axis=1)

    dx = df.torso_top_x - df.torso_bottom_x
    dy = df.torso_top_y - df.torso_bottom_y
    df['torso_tilt_signed'] = np.degrees(np.arctan2(dx, -dy))
    df['torso_tilt'] = (df.torso_tilt_signed.abs())
    df['status'] = np.select([df.leg_valid, df.arm_valid], ['ok', 'arm_ok'], default='partial')
    empty = df[[j + '_x' for j in POINTS if not j.startswith('torso')]].isna().all(axis=1)
    df.loc[empty, 'status'] = 'no_parts'
    return df
