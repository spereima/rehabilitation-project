import cv2
import numpy as np
import pandas as pd
from keypoints import extract_frame
from tracking import temporal_track
from geometry import angles_and_diagnostics

def process_video(video):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise OSError('Cannot open ' + str(video))

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Invalid FPS: ' + str(video))

    rows, i = [], 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for row in extract_frame(frame):
                row['frame'] = i
                row['time_s'] = i / fps
                rows.append(row)
            i += 1
    finally:
        cap.release()

    if not rows:
        return pd.DataFrame()

    df = pd.concat([angles_and_diagnostics(temporal_track(pd.DataFrame([r for r in rows if r['side'] == side])), fps)
        for side in ('left', 'right')], ignore_index=True)
    return df.sort_values(['frame', 'side']).reset_index(drop=True)
