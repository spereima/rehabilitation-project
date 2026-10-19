"""Analyze cam0-cam4 predictions for both modalities without retraining.
Run from rehabilitation-project root: python scripts/analyze_all_cameras.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = Path('output/project_name')
SPLIT = 'split_2'
RUN = 'full_16_exercises_cam0'
MODALITIES = ('sils', 'semantic_segmentation')
CAMERAS = ('cam0', 'cam1', 'cam2', 'cam3', 'cam4')
FPS = 30.0  # Verify source-video FPS before reporting time-based descriptors.
OUTPUT = Path('results/all_cameras_analysis')
PLOTS = OUTPUT / 'plots'


def load_predictions():
    chunks = []
    for modality in MODALITIES:
        base = ROOT / modality / SPLIT / RUN
        for camera in CAMERAS:
            path = (base / 'test_predictions.csv' if camera == 'cam0'
                    else base / 'cross_camera_tests' / camera / 'test_predictions.csv')
            if not path.exists():
                raise FileNotFoundError(f'Missing predictions: {path}')
            df = pd.read_csv(path)
            needed = {'subject', 'exercise', 'camera', 'frame_id', 'angle_gt', 'angle_pred'}
            if not needed.issubset(df.columns):
                raise ValueError(f'{path}: missing columns {needed - set(df.columns)}')
            df = df[list(needed)].copy()
            df['modality'] = modality
            df['camera'] = df['camera'].astype(str)
            if set(df['camera']) != {camera}:
                raise ValueError(f'{path}: unexpected cameras {df.camera.unique()}')
            df['subject'] = df['subject'].astype(int)
            df['exercise'] = df['exercise'].astype(int)
            df['frame_id'] = df['frame_id'].astype(int)
            df['angle_gt'] = pd.to_numeric(df['angle_gt'], errors='coerce')
            df['angle_pred'] = pd.to_numeric(df['angle_pred'], errors='coerce')
            df = df.dropna(subset=['angle_gt', 'angle_pred'])
            keys = ['subject', 'exercise', 'camera', 'frame_id']
            if df.duplicated(keys).any():
                raise ValueError(f'{path}: duplicate frame identifiers')
            chunks.append(df)
            print(f'{modality:23} {camera}: {len(df):6} frames')
    result = pd.concat(chunks, ignore_index=True)
    result['error_deg'] = result['angle_pred'] - result['angle_gt']
    result['absolute_error_deg'] = result['error_deg'].abs()
    return result


def error_summary(group):
    err = group['error_deg'].to_numpy(dtype=float)
    abs_err = np.abs(err)
    return pd.Series({
        'frames': len(group),
        'mae_deg': abs_err.mean(),
        'rmse_deg': np.sqrt(np.mean(err ** 2)),
        'bias_deg': err.mean(),
        'median_absolute_error_deg': np.median(abs_err),
        'p90_absolute_error_deg': np.percentile(abs_err, 90),
        'p95_absolute_error_deg': np.percentile(abs_err, 95),
        'within_5deg_pct': 100 * np.mean(abs_err <= 5),
        'within_10deg_pct': 100 * np.mean(abs_err <= 10),
        'within_15deg_pct': 100 * np.mean(abs_err <= 15),
    })


def descriptors(frames, angles):
    """Descriptors use actual frame-id differences, not adjacent-row assumptions."""
    frames = np.asarray(frames, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 3:
        return None
    dt = np.diff(frames) / FPS
    valid = dt > 0
    velocity = np.diff(angles)[valid] / dt[valid]
    if len(velocity) < 2:
        return None
    # Smoothness proxy = mean absolute change in angular velocity (deg/s).
    # Not a dimensionally normalized jerk or clinical smoothness metric.
    return {
        'rom_deg': float(np.max(angles) - np.min(angles)),
        'mean_velocity_deg_s': float(np.mean(np.abs(velocity))),
        'peak_velocity_deg_s': float(np.max(np.abs(velocity))),
        'smoothness_proxy': float(np.mean(np.abs(np.diff(velocity)))),
    }


def detect_repetitions(frames, gt):
    """Heuristic GT trough-to-trough intervals; gaps split sequences."""
    if len(gt) < 90:
        return []
    smoothed = pd.Series(gt).rolling(9, center=True, min_periods=1).mean().to_numpy()
    troughs, _ = find_peaks(-smoothed, distance=45, prominence=10)
    boundaries = []
    for start, end in zip(troughs[:-1], troughs[1:]):
        if end - start < 15:
            continue
        if np.max(np.diff(frames[start:end + 1])) > 1:
            continue
        boundaries.append((int(start), int(end)))
    return boundaries


def analyze_sequences(df):
    seq_rows, rep_rows, first_last_rows = [], [], []
    grouped = df.groupby(['modality', 'subject', 'exercise', 'camera'], sort=True)
    for (modality, subject, exercise, camera), group in grouped:
        group = group.sort_values('frame_id')
        frames = group.frame_id.to_numpy()
        gt = group.angle_gt.to_numpy()
        pred = group.angle_pred.to_numpy()
        gt_d = descriptors(frames, gt)
        pred_d = descriptors(frames, pred)
        if gt_d is None or pred_d is None:
            continue
        base = {'modality': modality, 'subject': subject,
                'exercise': exercise, 'camera': camera, 'frames': len(group)}
        record = dict(base)
        record.update({f'gt_{key}': value for key, value in gt_d.items()})
        record.update({f'pred_{key}': value for key, value in pred_d.items()})
        for key in gt_d:
            record[f'{key}_absolute_error'] = abs(gt_d[key] - pred_d[key])
        record['angle_mae_deg'] = float(np.mean(np.abs(gt - pred)))
        seq_rows.append(record)
        local_reps = []
        for number, (start, end) in enumerate(detect_repetitions(frames, gt), 1):
            rep_gt = descriptors(frames[start:end + 1], gt[start:end + 1])
            rep_pred = descriptors(frames[start:end + 1], pred[start:end + 1])
            if rep_gt is None or rep_pred is None:
                continue
            rep = {**base, 'repetition': number,
                   'start_frame': int(frames[start]), 'end_frame': int(frames[end]),
                   'duration_s': float((frames[end] - frames[start]) / FPS),
                   'angle_mae_deg': float(np.mean(np.abs(gt[start:end + 1] - pred[start:end + 1])))}
            for key in rep_gt:
                rep[f'gt_{key}'] = rep_gt[key]
                rep[f'pred_{key}'] = rep_pred[key]
                rep[f'{key}_absolute_error'] = abs(rep_gt[key] - rep_pred[key])
            rep_rows.append(rep)
            local_reps.append(rep)
        if len(local_reps) >= 2:
            first_last_rows.append({**local_reps[0], 'position': 'first'})
            first_last_rows.append({**local_reps[-1], 'position': 'last'})
    return pd.DataFrame(seq_rows), pd.DataFrame(rep_rows), pd.DataFrame(first_last_rows)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def make_plots(df, camera_metrics, exercise_metrics, sequences):
    PLOTS.mkdir(parents=True, exist_ok=True)
    for metric in ('mae_deg', 'rmse_deg', 'bias_deg'):
        pivot = camera_metrics.pivot(index='camera', columns='modality', values=metric).reindex(CAMERAS)
        pivot.plot(kind='bar', figsize=(10, 5), rot=0)
        plt.ylabel(metric)
        plt.title(f'Camera comparison: {metric}')
        save_figure(PLOTS / f'cameras_{metric}.png')
    pivot = exercise_metrics.pivot_table(index='exercise', columns=['modality', 'camera'], values='mae_deg')
    for modality in MODALITIES:
        values = pivot[modality].reindex(columns=CAMERAS)
        plt.figure(figsize=(11, 7))
        plt.imshow(values.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='MAE (degrees)')
        plt.xticks(range(len(values)), [f'{int(x):02d}' for x in values.index])
        plt.yticks(range(len(CAMERAS)), CAMERAS)
        plt.xlabel('Exercise')
        plt.ylabel('Camera')
        plt.title(f'Exercise × camera MAE: {modality}')
        save_figure(PLOTS / f'heatmap_{modality}.png')
    for modality in MODALITIES:
        subset = df[df.modality == modality]
        plt.figure(figsize=(8, 5))
        for camera in CAMERAS:
            values = subset.loc[subset.camera == camera, 'absolute_error_deg']
            plt.hist(values, bins=np.arange(0, 91, 3), density=True,
                     alpha=0.30, label=camera)
        plt.xlim(0, 90)
        plt.xlabel('Absolute angle error (degrees)')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Error distribution: {modality}')
        save_figure(PLOTS / f'error_distribution_{modality}.png')
    # Per-sequence GT vs predicted plots, all available test sequences.
    count = 0
    for (modality, subject, exercise, camera), group in df.groupby(
            ['modality', 'subject', 'exercise', 'camera']):
        group = group.sort_values('frame_id')
        if len(group) < 3:
            continue
        directory = PLOTS / 'ground_truth_vs_predicted' / modality / camera
        directory.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 4))
        time = group.frame_id.to_numpy() / FPS
        plt.plot(time, group.angle_gt, label='Ground truth', linewidth=1.3)
        plt.plot(time, group.angle_pred, label='Predicted', linewidth=1.0, alpha=0.85)
        plt.xlabel('Time (s; assuming 30 FPS)')
        plt.ylabel('Angle (degrees)')
        mae = group.absolute_error_deg.mean()
        plt.title(f'{modality} | {camera} | subject {subject} | exercise {exercise:02d} | MAE {mae:.2f}°')
        plt.legend()
        save_figure(directory / f'subject_{subject}_exercise_{exercise:02d}.png')
        count += 1
    print(f'Saved {count} GT vs predicted sequence plots')
    if not sequences.empty:
        for metric in ('rom_deg_absolute_error', 'mean_velocity_deg_s_absolute_error',
                       'peak_velocity_deg_s_absolute_error', 'smoothness_proxy_absolute_error'):
            if metric not in sequences:
                continue
            summary = sequences.groupby(['modality', 'camera'])[metric].mean().unstack('modality').reindex(CAMERAS)
            summary.plot(kind='bar', figsize=(10, 5), rot=0)
            plt.ylabel(metric)
            plt.title(f'Movement descriptor error: {metric}')
            save_figure(PLOTS / f'descriptor_{metric}.png')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    df.to_csv(OUTPUT / 'all_predictions.csv', index=False)
    camera = df.groupby(['modality', 'camera']).apply(error_summary).reset_index()
    exercise = df.groupby(['modality', 'camera', 'exercise']).apply(error_summary).reset_index()
    subject = df.groupby(['modality', 'camera', 'subject']).apply(error_summary).reset_index()
    camera.to_csv(OUTPUT / 'metrics_by_camera.csv', index=False)
    exercise.to_csv(OUTPUT / 'metrics_by_exercise.csv', index=False)
    subject.to_csv(OUTPUT / 'metrics_by_subject.csv', index=False)
    sequences, repetitions, first_last = analyze_sequences(df)
    sequences.to_csv(OUTPUT / 'movement_descriptors.csv', index=False)
    repetitions.to_csv(OUTPUT / 'repetition_descriptors.csv', index=False)
    first_last.to_csv(OUTPUT / 'first_vs_last_repetition.csv', index=False)
    if not sequences.empty:
        sequences.groupby(['modality', 'camera', 'exercise'], as_index=False)[
            ['angle_mae_deg', 'rom_deg_absolute_error', 'mean_velocity_deg_s_absolute_error',
             'peak_velocity_deg_s_absolute_error', 'smoothness_proxy_absolute_error']
        ].mean().to_csv(OUTPUT / 'movement_summary_by_exercise.csv', index=False)
    make_plots(df, camera, exercise, sequences)
    print('\nCamera metrics:')
    print(camera[['modality', 'camera', 'frames', 'mae_deg', 'rmse_deg', 'bias_deg']].to_string(index=False))
    print(f'\nSaved results to {OUTPUT.resolve()}')
    print('NOTE: Verify 30 FPS and camera/frame synchronization; repetition boundaries are heuristic.')


if __name__ == '__main__':
    main()
