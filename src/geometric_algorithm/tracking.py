import numpy as np
import pandas as pd
from config import (POINTS, MAX_JUMP_REL_SCALE, MAX_GAP_FRAMES)

def coords(df, name):
    return df[[name + '_x', name + '_y']].to_numpy(float)

def temporal_track(df):
    df = df.copy().reset_index(drop=True)
    scale = max(25.0, float(df.body_scale.median()))

    for j in POINTS:
        a = coords(df, j)
        original = np.isfinite(a).all(axis=1)
        status = df[j + '_status'].to_numpy(object).copy()

        if len(a) >= 5:
            median = pd.DataFrame(a).rolling(5, center=True, min_periods=3).median().to_numpy()
            bad = (original & np.isfinite(median).all(axis=1) &(np.linalg.norm(a - median, axis=1) > MAX_JUMP_REL_SCALE * scale))
            a[bad] = np.nan
            status[bad] = 'temporal_outlier'

        good = np.flatnonzero(np.isfinite(a).all(axis=1))
        for lo, hi in zip(good[:-1], good[1:]):
            gap = hi - lo - 1

            if not 0 < gap <= MAX_GAP_FRAMES:
                continue
            if (np.linalg.norm(a[hi] - a[lo]) > MAX_JUMP_REL_SCALE * scale * (gap + 1)):
                continue

            for i in range(lo + 1, hi):
                a[i] = (a[lo] + (a[hi] - a[lo]) * (i - lo) / (hi - lo))
                status[i] = 'estimated'

        for k, ax in enumerate('xy'):
            df[j + '_' + ax] = a[:, k]
        df[j + '_status'] = status
        df[j + '_confidence'] = np.select([status == 'observed', status == 'estimated'], [0.95, 0.55], default=0.0)

    return df