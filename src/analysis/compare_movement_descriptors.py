
"""Compare movement descriptors using final held-out test predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PREDICTION_FILES = {
    "sils": Path(
        "output/ucophyrehab_final/test_results/sils/all_predictions.csv"
    ),
    "semantic_segmentation": Path(
        "output/ucophyrehab_final/test_results/"
        "semantic_segmentation/all_predictions.csv"
    ),
}

OUTPUT_DIR = Path("results/final_test/movement_descriptors")

FPS = 30
MIN_REPETITION_SECONDS = 1.5
MIN_PROMINENCE_DEGREES = 10
SMOOTHING_WINDOW = 9


def calculate_descriptors(frames, angles):
    """Calculate descriptors with frame-aware time differences."""

    frames = np.asarray(frames, dtype=float)
    angles = np.asarray(angles, dtype=float)

    valid = np.isfinite(frames) & np.isfinite(angles)

    frames = frames[valid]
    angles = angles[valid]

    if len(angles) < 3:
        return None

    frame_deltas = np.diff(frames)

    if np.any(frame_deltas <= 0):
        return None

    dt = frame_deltas / FPS

    velocity = np.diff(angles) / dt

    velocity_time = (
        frames[:-1] + frames[1:]
    ) / (2 * FPS)

    acceleration_dt = np.diff(velocity_time)

    acceleration = (
        np.diff(velocity) / acceleration_dt
    )

    return {
        "frames": len(angles),
        "rom_deg": float(np.ptp(angles)),
        "mean_velocity_deg_s": float(
            np.mean(np.abs(velocity))
        ),
        "peak_velocity_deg_s": float(
            np.max(np.abs(velocity))
        ),
        "smoothness_proxy": float(
            np.mean(np.abs(acceleration))
        ),
    }


def detect_boundaries(frames, gt_angles):
    """Detect heuristic GT trough-to-trough repetitions."""

    series = pd.Series(gt_angles, dtype=float)

    smoothed = series.rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    ).mean().to_numpy()

    minima, _ = find_peaks(
        -smoothed,
        distance=max(
            1,
            int(MIN_REPETITION_SECONDS * FPS)
        ),
        prominence=MIN_PROMINENCE_DEGREES
    )

    return [
        (int(start), int(end))
        for start, end in zip(
            minima[:-1],
            minima[1:]
        )
    ]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_descriptors = []
    all_repetitions = []
    all_first_last = []

    for modality, path in PREDICTION_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        data = pd.read_csv(
            path,
            dtype={"subject": str}
        )

        data = data.rename(
            columns={
                "ground_truth_angle": "angle_gt",
                "predicted_angle": "angle_pred"
            }
        )

        required = {
            "subject",
            "exercise",
            "camera",
            "frame_id",
            "angle_gt",
            "angle_pred"
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        for (subject, exercise, camera), group in data.groupby(
            ["subject", "exercise", "camera"],
            sort=True
        ):

            group = (
                group.sort_values("frame_id")
                .drop_duplicates("frame_id")
                .dropna(
                    subset=[
                        "angle_gt",
                        "angle_pred",
                        "frame_id"
                    ]
                )
            )

            frames = group["frame_id"].to_numpy()

            gt = group["angle_gt"].to_numpy(
                dtype=float
            )

            pred = group["angle_pred"].to_numpy(
                dtype=float
            )

            gt_metrics = calculate_descriptors(
                frames, gt
            )

            pred_metrics = calculate_descriptors(
                frames, pred
            )

            if gt_metrics is None or pred_metrics is None:
                continue

            base = {
                "modality": modality,
                "subject": subject,
                "exercise": exercise,
                "camera": camera
            }

            record = dict(base)

            record.update({
                f"gt_{key}": value
                for key, value in gt_metrics.items()
            })

            record.update({
                f"pred_{key}": value
                for key, value in pred_metrics.items()
            })

            record["rom_absolute_error_deg"] = abs(
                gt_metrics["rom_deg"]
                - pred_metrics["rom_deg"]
            )

            record["angle_mae_deg"] = float(
                np.mean(np.abs(gt - pred))
            )

            all_descriptors.append(record)

            boundaries = detect_boundaries(
                frames, gt
            )

            group_repetitions = []

            for number, (start_idx, end_idx) in enumerate(
                boundaries,
                start=1
            ):

                frame_slice = frames[
                    start_idx:end_idx + 1
                ]

                gt_slice = gt[
                    start_idx:end_idx + 1
                ]

                pred_slice = pred[
                    start_idx:end_idx + 1
                ]

                gt_rep = calculate_descriptors(
                    frame_slice,
                    gt_slice
                )

                pred_rep = calculate_descriptors(
                    frame_slice,
                    pred_slice
                )

                if gt_rep is None or pred_rep is None:
                    continue

                rep = {
                    **base,
                    "repetition": number,
                    "start_frame": int(frame_slice[0]),
                    "end_frame": int(frame_slice[-1]),
                    "duration_s": float(
                        (frame_slice[-1] - frame_slice[0])
                        / FPS
                    ),
                    "gt_rom_deg": gt_rep["rom_deg"],
                    "pred_rom_deg": pred_rep["rom_deg"],
                    "rom_absolute_error_deg": abs(
                        gt_rep["rom_deg"]
                        - pred_rep["rom_deg"]
                    ),
                    "gt_peak_velocity_deg_s":
                        gt_rep["peak_velocity_deg_s"],
                    "pred_peak_velocity_deg_s":
                        pred_rep["peak_velocity_deg_s"],
                    "gt_smoothness_proxy":
                        gt_rep["smoothness_proxy"],
                    "pred_smoothness_proxy":
                        pred_rep["smoothness_proxy"],
                }

                group_repetitions.append(rep)
                all_repetitions.append(rep)

            if len(group_repetitions) >= 2:
                all_first_last.extend([
                    group_repetitions[0],
                    group_repetitions[-1]
                ])

            print(
                f"{modality}: "
                f"subject={subject}, "
                f"exercise={exercise}, "
                f"camera={camera}; "
                f"MAE={record['angle_mae_deg']:.2f}°, "
                f"ROM error="
                f"{record['rom_absolute_error_deg']:.2f}°, "
                f"repetitions={len(group_repetitions)}"
            )

    outputs = {
        "movement_descriptors.csv": all_descriptors,
        "repetition_descriptors.csv": all_repetitions,
        "first_vs_last_repetition.csv": all_first_last
    }

    for filename, records in outputs.items():

        if records:

            pd.DataFrame(records).to_csv(
                OUTPUT_DIR / filename,
                index=False
            )

            print("Saved:", OUTPUT_DIR / filename)

    if all_descriptors:

        summary = (
            pd.DataFrame(all_descriptors)
            .groupby(
                ["modality", "exercise"],
                as_index=False
            )[
                [
                    "angle_mae_deg",
                    "rom_absolute_error_deg"
                ]
            ]
            .mean()
        )

        summary.to_csv(
            OUTPUT_DIR / "summary_by_exercise.csv",
            index=False
        )

        print("\nSummary by exercise:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
