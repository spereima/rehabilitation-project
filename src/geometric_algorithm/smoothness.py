import numpy as np

def smoothness(angle, fps):
    x = np.asarray(angle, dtype=float)
    if len(x) < 7 or not np.isfinite(x).all():
        return np.nan

    acceleration = np.gradient(np.gradient(x, 1 / fps), 1 / fps)
    return float(np.sqrt(np.mean(acceleration ** 2)))