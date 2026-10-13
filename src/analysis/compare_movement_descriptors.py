"""Batch movement descriptors for test predictions from two image modalities.

Run from the rehabilitation-project root:
    python -m src.analysis.compare_movement_descriptors
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PREDICTION_FILES = {
    "sils": Path("output/project_name/sils/split_2/full_sils/test_predictions.csv"),
    "semantic_segmentation": Path(
        "output/project_name/semantic_segmentation/split_2/full_semantic/test_predictions.csv"
    ),
}
OUTPUT_DIR = Path("results/full_dataset/movement_descriptors")
FPS = 30
MIN_REPETITION_SECONDS = 1.5
MIN_PROMINENCE_DEGREES = 10
SMOOTHING_WINDOW = 9


def calculate_descriptors(frames, angles):
    """Return whole-sequence metrics; velocities respect frame-number gaps."""
    angles = np.asarray(angles, dtype=float)
    frames = np.asarray(frames, dtype=float)
    valid = np.isfinite(angles) & np.isfinite(frames)
    angles, frames = angles[valid], frames[valid]
    if len(angles) < 3:
        return None

    frame_deltas = np.diff(frames)
    if np.any(frame_deltas <= 0):
        return None
    dt = frame_deltas / FPS
    velocity = np.diff(angles) / dt
    acceleration_proxy = np.diff(velocity)
    return {
        "frames": len(angles),
        "rom_deg": float(np.ptp(angles)),
        "mean_velocity_deg_s": float(np.mean(np.abs(velocity))),
        "peak_velocity_deg_s": float(np.max(np.abs(velocity))),
        "smoothness_proxy": (
            float(np.mean(np.abs(acceleration_proxy)))
            if len(acceleration_proxy) else float("nan")
        ),
    }


def detect_boundaries(frames, gt_angles):
    """Heuristic GT trough-to-trough segmentation; review per exercise."""
    series = pd.Series(gt_angles, dtype=float)
    smoothed = series.rolling(
        window=SMOOTHING_WINDOW, center=True, min_periods=1
    ).mean().to_numpy()
    min_distance = max(1, int(MIN_REPETITION_SECONDS * FPS))
    minima, _ = find_peaks(
        -smoothed,
        distance=min_distance,
        prominence=MIN_PROMINENCE_DEGREES,
    )
    return [(int(start), int(end)) for start, end in zip(minima[:-1], minima[1:])]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_descriptors = []
    all_repetitions = []
    all_first_last = []

    for modality, path in PREDICTION_FILES.items():
        if not path.exists():
            print(f"Missing {modality}: {path}")
            continue

        data = pd.read_csv(path)
        required = {"subject", "exercise", "camera", "frame_id", "angle_gt", "angle_pred"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for (subject, exercise, camera), group in data.groupby(
            ["subject", "exercise", "camera"], sort=True
        ):
            group = group.sort_values("frame_id").drop_duplicates("frame_id")
            group = group.dropna(subset=["angle_gt", "angle_pred", "frame_id"])
            frames = group["frame_id"].to_numpy()
            gt = group["angle_gt"].to_numpy(dtype=float)
            pred = group["angle_pred"].to_numpy(dtype=float)
            gt_metrics = calculate_descriptors(frames, gt)
            pred_metrics = calculate_descriptors(frames, pred)
            if gt_metrics is None or pred_metrics is None:
                print(f"Skipping short sequence: {modality}, {subject}, {exercise}, {camera}")
                continue

            base = {
                "modality": modality,
                "subject": subject,
                "exercise": exercise,
                "camera": camera,
            }
            record = dict(base)
            record.update({f"gt_{key}": value for key, value in gt_metrics.items()})
            record.update({f"pred_{key}": value for key, value in pred_metrics.items()})
            record["rom_absolute_error_deg"] = abs(
                gt_metrics["rom_deg"] - pred_metrics["rom_deg"]
            )
            record["angle_mae_deg"] = float(np.mean(np.abs(gt - pred)))
            all_descriptors.append(record)

            # Repetition boundaries are found from GT, then applied to both
            # modalities' predictions. They are heuristic, not annotations.
            boundaries = detect_boundaries(frames, gt)
            group_repetitions = []
            for number, (start_idx, end_idx) in enumerate(boundaries, start=1):
                frame_slice = frames[start_idx:end_idx + 1]
                gt_slice = gt[start_idx:end_idx + 1]
                pred_slice = pred[start_idx:end_idx + 1]
                gt_rep = calculate_descriptors(frame_slice, gt_slice)
                pred_rep = calculate_descriptors(frame_slice, pred_slice)
                if gt_rep is None or pred_rep is None:
                    continue
                rep = {
                    **base,
                    "repetition": number,
                    "start_frame": int(frame_slice[0]),
                    "end_frame": int(frame_slice[-1]),
                    "duration_s": float((frame_slice[-1] - frame_slice[0]) / FPS),
                    "gt_rom_deg": gt_rep["rom_deg"],
                    "pred_rom_deg": pred_rep["rom_deg"],
                    "rom_absolute_error_deg": abs(gt_rep["rom_deg"] - pred_rep["rom_deg"]),
                    "gt_peak_velocity_deg_s": gt_rep["peak_velocity_deg_s"],
                    "pred_peak_velocity_deg_s": pred_rep["peak_velocity_deg_s"],
                    "gt_smoothness_proxy": gt_rep["smoothness_proxy"],
                    "pred_smoothness_proxy": pred_rep["smoothness_proxy"],
                }
                group_repetitions.append(rep)
                all_repetitions.append(rep)

            if len(group_repetitions) >= 2:
                all_first_last.extend([group_repetitions[0], group_repetitions[-1]])

            print(
                f"{modality}: subject={subject}, exercise={exercise}, camera={camera}; "
                f"MAE={record['angle_mae_deg']:.2f}°, "
                f"ROM error={record['rom_absolute_error_deg']:.2f}°, "
                f"detected repetitions={len(group_repetitions)}"
            )

    outputs = {
        "movement_descriptors.csv": all_descriptors,
        "repetition_descriptors.csv": all_repetitions,
        "first_vs_last_repetition.csv": all_first_last,
    }
    for filename, records in outputs.items():
        if records:
            pd.DataFrame(records).to_csv(OUTPUT_DIR / filename, index=False)
            print(f"Saved: {OUTPUT_DIR / filename}")
        else:
            print(f"No rows for {filename}")

    if all_descriptors:
        summary = (
            pd.DataFrame(all_descriptors)
            .groupby(["modality", "exercise"], as_index=False)
            [["angle_mae_deg", "rom_absolute_error_deg"]]
            .mean()
        )
        summary.to_csv(OUTPUT_DIR / "summary_by_exercise.csv", index=False)
        print("\nSummary by exercise:")
        print(summary.to_string(index=False))

    print("\nNote: repetition boundaries are heuristic GT troughs; inspect before reporting repetition-level results.")


if __name__ == "__main__":
    main()
