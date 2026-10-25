
"""Compare final predictions across all test participants."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Experiment settings
points = 100

# Input files
results_root = Path(
    "/content/drive/MyDrive/rehabilitation-project-data"
    "/cross_validation/output/ucophyrehab_final/test_results"
)

sils_path = results_root / "sils" / "all_predictions.csv"

segmentation_path = (
    results_root
    / "semantic_segmentation"
    / "all_predictions.csv"
)

output_dir = Path("results/final_test/modality_comparison")

output_dir.mkdir(parents=True, exist_ok=True)


# Load predictions
sils = pd.read_csv(
    sils_path,
    dtype={"subject": str}
)

segmentation = pd.read_csv(
    segmentation_path,
    dtype={"subject": str}
)

sils = sils.rename(
    columns={
        "ground_truth_angle": "angle_gt",
        "predicted_angle": "angle_pred"
    }
)

segmentation = segmentation.rename(
    columns={
        "ground_truth_angle": "angle_gt",
        "predicted_angle": "angle_pred"
    }
)


# Match frames across modalities
keys = [
    "subject",
    "exercise",
    "camera",
    "frame_id"
]

required = keys + ["angle_gt", "angle_pred"]

for modality_name, data in [
    ("SILS", sils),
    ("Semantic Segmentation", segmentation)
]:
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"{modality_name}: missing columns {sorted(missing)}"
        )

    if data.duplicated(keys).any():
        raise ValueError(
            f"{modality_name}: duplicate frame identifiers"
        )

sils = sils.dropna(subset=required)
segmentation = segmentation.dropna(subset=required)

comparison = pd.merge(
    sils[required],
    segmentation[required],
    on=keys,
    suffixes=("_sils", "_segmentation"),
    validate="one_to_one"
)

if comparison.empty:
    raise ValueError("No matching frames found.")

if not np.allclose(
    comparison["angle_gt_sils"],
    comparison["angle_gt_segmentation"],
    atol=1e-5,
    rtol=0
):
    raise ValueError("Ground Truth differs between modalities.")

comparison = comparison.rename(
    columns={
        "angle_gt_sils": "angle_gt",
        "angle_pred_sils": "angle_pred_sils",
        "angle_pred_segmentation": "angle_pred_segmentation"
    }
)

comparison = comparison.drop(
    columns=["angle_gt_segmentation"]
)

comparison = comparison.sort_values(keys).reset_index(drop=True)

comparison["error_sils"] = (
    comparison["angle_pred_sils"] - comparison["angle_gt"]
).abs()

comparison["error_segmentation"] = (
    comparison["angle_pred_segmentation"] - comparison["angle_gt"]
).abs()

comparison.to_csv(
    output_dir / "frame_comparison.csv",
    index=False
)


# Calculate movement descriptors
def calculate_descriptors(data, angle_column):
    angle = data[angle_column].to_numpy(dtype=float)

    return {
        "rom_deg": float(np.ptp(angle))
    }


# Overall comparison
results = []

modalities = [
    ("Ground Truth", "angle_gt"),
    ("SILS", "angle_pred_sils"),
    ("Semantic Segmentation", "angle_pred_segmentation")
]

sequence_results = []
trajectory_results = []

for (subject, exercise, camera), group in comparison.groupby(
    ["subject", "exercise", "camera"],
    sort=True
):
    group = group.sort_values("frame_id")

    if len(group) < 3:
        continue

    frames = group["frame_id"].to_numpy(dtype=float)

    if frames[-1] == frames[0]:
        continue

    normalized_time = (
        (frames - frames[0])
        / (frames[-1] - frames[0])
        * 100
    )

    common_time = np.linspace(0, 100, points)

    sequence_record = {
        "subject": subject,
        "exercise": exercise,
        "camera": camera,
        "frames": len(group)
    }

    for modality_name, angle_column in modalities:
        descriptors = calculate_descriptors(
            group,
            angle_column
        )

        sequence_record[
            f"rom_{angle_column}"
        ] = descriptors["rom_deg"]

        if modality_name != "Ground Truth":
            errors = (
                group[angle_column] - group["angle_gt"]
            )

            sequence_record[
                f"mae_{angle_column}"
            ] = errors.abs().mean()

            sequence_record[
                f"rmse_{angle_column}"
            ] = np.sqrt((errors ** 2).mean())

    sequence_results.append(sequence_record)

    gt = np.interp(
        common_time,
        normalized_time,
        group["angle_gt"]
    )

    pred_sils = np.interp(
        common_time,
        normalized_time,
        group["angle_pred_sils"]
    )

    pred_segmentation = np.interp(
        common_time,
        normalized_time,
        group["angle_pred_segmentation"]
    )

    error_sils = np.interp(
        common_time,
        normalized_time,
        group["error_sils"]
    )

    error_segmentation = np.interp(
        common_time,
        normalized_time,
        group["error_segmentation"]
    )

    for index in range(points):
        trajectory_results.append({
            "subject": subject,
            "exercise": exercise,
            "camera": camera,
            "time_percent": common_time[index],
            "angle_gt": gt[index],
            "angle_pred_sils": pred_sils[index],
            "angle_pred_segmentation": pred_segmentation[index],
            "error_sils": error_sils[index],
            "error_segmentation": error_segmentation[index]
        })


sequence_df = pd.DataFrame(sequence_results)

trajectory_df = pd.DataFrame(trajectory_results)

if sequence_df.empty:
    raise ValueError("No valid test sequences found.")

sequence_df.to_csv(
    output_dir / "sequence_comparison.csv",
    index=False
)

trajectory_df.to_csv(
    output_dir / "trajectory_comparison.csv",
    index=False
)


# Average across sequences
mean_trajectories = (
    trajectory_df.groupby("time_percent", as_index=False)[
        [
            "angle_gt",
            "angle_pred_sils",
            "angle_pred_segmentation",
            "error_sils",
            "error_segmentation"
        ]
    ]
    .mean()
)

mean_trajectories.to_csv(
    output_dir / "mean_trajectories.csv",
    index=False
)


# Overall metrics
for modality_name, angle_column in modalities:

    rom = sequence_df[
        f"rom_{angle_column}"
    ].mean()

    if modality_name == "Ground Truth":
        mae = 0.0
        rmse = 0.0
    else:
        errors = (
            comparison[angle_column] - comparison["angle_gt"]
        )

        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())

    results.append({
        "modality": modality_name,
        "mae_deg": mae,
        "rmse_deg": rmse,
        "mean_rom_deg": rom
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    output_dir / "modality_comparison.csv",
    index=False
)

print("\nOverall modality comparison:")
print(results_df.round(2).to_string(index=False))

print("\nNumber of participants:")
print(sequence_df["subject"].nunique())

print("\nNumber of sequences:")
print(len(sequence_df))


# Plot 1: Ground Truth vs both modalities
plt.figure(figsize=(14, 6))

plt.plot(
    mean_trajectories["time_percent"],
    mean_trajectories["angle_gt"],
    label="Ground Truth",
    linewidth=2
)

plt.plot(
    mean_trajectories["time_percent"],
    mean_trajectories["angle_pred_sils"],
    label="SILS",
    alpha=0.8
)

plt.plot(
    mean_trajectories["time_percent"],
    mean_trajectories["angle_pred_segmentation"],
    label="Semantic Segmentation",
    alpha=0.8
)

plt.xlabel("Normalized Video Time (%)")
plt.ylabel("Mean Angle (degrees)")
plt.title(
    "Mean Ground Truth vs SILS vs Semantic Segmentation"
)
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    output_dir / "predictions_comparison.png",
    dpi=300
)

plt.close()


# Plot 2: Absolute errors over time
plt.figure(figsize=(14, 6))

plt.plot(
    mean_trajectories["time_percent"],
    mean_trajectories["error_sils"],
    label="SILS Absolute Error"
)

plt.plot(
    mean_trajectories["time_percent"],
    mean_trajectories["error_segmentation"],
    label="Segmentation Absolute Error"
)

plt.xlabel("Normalized Video Time (%)")
plt.ylabel("Mean Absolute Error (degrees)")
plt.title("Mean Prediction Errors Over Time")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    output_dir / "errors_comparison.png",
    dpi=300
)

plt.close()


# Plot 3: ROM comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["modality"],
    results_df["mean_rom_deg"]
)

plt.ylabel("Mean Range of Motion (degrees)")
plt.title("Mean ROM Comparison Across All Test Sequences")
plt.tight_layout()

plt.savefig(
    output_dir / "rom_comparison.png",
    dpi=300
)

plt.close()


print(f"\nResults saved to: {output_dir}")
