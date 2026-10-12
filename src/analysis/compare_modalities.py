
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Experiment settings
exercise = 1
subject = 1
camera = "cam0"
fps = 30

repetition_boundaries = [
    (5, 311),
    (311, 533),
    (533, 733),
    (733, 910),
]

# Input files
sils_path = (
    "output/ucophyrehapp_sample/sils/split_1/"
    "sils_split1_01_05_cam0/test_predictions.csv"
)

segmentation_path = (
    "output/ucophyrehapp_sample/semantic_segmentation/split_1/"
    "semantic_segmentation_split1_01_05_cam0/test_predictions.csv"
)

output_dir = "results/modality_comparison/ex01_subject1_cam0"

os.makedirs(output_dir, exist_ok=True)


# Load predictions
sils = pd.read_csv(sils_path)
segmentation = pd.read_csv(segmentation_path)


def select_sequence(data):
    sequence = data[
        (data["exercise"] == exercise)
        & (data["subject"] == subject)
        & (data["camera"] == camera)
    ].copy()

    sequence = sequence.sort_values("frame_id")

    return sequence.reset_index(drop=True)


sils = select_sequence(sils)
segmentation = select_sequence(segmentation)

if sils.empty or segmentation.empty:
    raise ValueError("No predictions found for the selected sequence.")


# Match the same frames across modalities
comparison = pd.merge(
    sils[["frame_id", "angle_gt", "angle_pred"]],
    segmentation[["frame_id", "angle_gt", "angle_pred"]],
    on="frame_id",
    suffixes=("_sils", "_segmentation"),
    validate="one_to_one",
)

if len(comparison) != len(sils) or len(comparison) != len(segmentation):
    raise ValueError("The modalities do not contain the same frames.")

if not np.allclose(
    comparison["angle_gt_sils"],
    comparison["angle_gt_segmentation"],
    atol=1e-5,
    rtol=0,
):
    raise ValueError("Ground Truth differs between modalities.")

comparison = comparison.rename(
    columns={
        "angle_gt_sils": "angle_gt",
        "angle_pred_sils": "angle_pred_sils",
        "angle_pred_segmentation": "angle_pred_segmentation",
    }
)

comparison = comparison.drop(
    columns=["angle_gt_segmentation"]
)

comparison = comparison.sort_values("frame_id").reset_index(drop=True)

comparison["time_s"] = (
    comparison["frame_id"] - comparison["frame_id"].iloc[0]
) / fps

comparison["error_sils"] = (
    comparison["angle_pred_sils"] - comparison["angle_gt"]
).abs()

comparison["error_segmentation"] = (
    comparison["angle_pred_segmentation"] - comparison["angle_gt"]
).abs()

comparison.to_csv(
    os.path.join(output_dir, "frame_comparison.csv"),
    index=False,
)


# Calculate movement descriptors
def calculate_descriptors(data, angle_column):
    angle = data[angle_column]

    frame_difference = data["frame_id"].diff()
    time_difference = frame_difference / fps

    angular_velocity = angle.diff() / time_difference

    velocity_change = angular_velocity.diff().abs()

    return {
        "rom_deg": angle.max() - angle.min(),
        "mean_velocity_deg_s": angular_velocity.abs().mean(),
        "peak_velocity_deg_s": angular_velocity.abs().max(),
        "smoothness_proxy": velocity_change.mean(),
    }


# Overall comparison
results = []

modalities = [
    ("Ground Truth", "angle_gt"),
    ("Sils", "angle_pred_sils"),
    ("Semantic Segmentation", "angle_pred_segmentation"),
]

for modality_name, angle_column in modalities:
    descriptors = calculate_descriptors(
        comparison,
        angle_column,
    )

    if modality_name == "Ground Truth":
        mae = 0.0
        rmse = 0.0
    else:
        errors = comparison[angle_column] - comparison["angle_gt"]

        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())

    results.append(
        {
            "modality": modality_name,
            "mae_deg": mae,
            "rmse_deg": rmse,
            **descriptors,
        }
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(output_dir, "modality_comparison.csv"),
    index=False,
)

print("\nOverall modality comparison:")
print(results_df.round(2).to_string(index=False))


# Repetition comparison
repetition_results = []

for repetition_number, boundaries in enumerate(
    repetition_boundaries,
    start=1,
):
    start_frame, end_frame = boundaries

    repetition = comparison[
        (comparison["frame_id"] >= start_frame)
        & (comparison["frame_id"] <= end_frame)
    ].copy()

    if len(repetition) < 3:
        raise ValueError(
            f"Not enough frames in repetition {repetition_number}."
        )

    for modality_name, angle_column in modalities:
        descriptors = calculate_descriptors(
            repetition,
            angle_column,
        )

        mae = (
            repetition[angle_column] - repetition["angle_gt"]
        ).abs().mean()

        repetition_results.append(
            {
                "repetition": repetition_number,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "modality": modality_name,
                "mae_deg": mae,
                **descriptors,
            }
        )

repetition_df = pd.DataFrame(repetition_results)

repetition_df.to_csv(
    os.path.join(output_dir, "repetition_comparison.csv"),
    index=False,
)

print("\nRepetition comparison:")
print(repetition_df.round(2).to_string(index=False))


# Plot 1: Ground Truth vs both modalities
plt.figure(figsize=(14, 6))

plt.plot(
    comparison["time_s"],
    comparison["angle_gt"],
    label="Ground Truth",
    linewidth=2,
)

plt.plot(
    comparison["time_s"],
    comparison["angle_pred_sils"],
    label="Sils",
    alpha=0.8,
)

plt.plot(
    comparison["time_s"],
    comparison["angle_pred_segmentation"],
    label="Semantic Segmentation",
    alpha=0.8,
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Ground Truth vs Sils vs Semantic Segmentation")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "predictions_comparison.png"),
    dpi=300,
)

plt.close()


# Plot 2: Absolute errors over time
plt.figure(figsize=(14, 6))

plt.plot(
    comparison["time_s"],
    comparison["error_sils"],
    label="Sils Absolute Error",
)

plt.plot(
    comparison["time_s"],
    comparison["error_segmentation"],
    label="Segmentation Absolute Error",
)

plt.xlabel("Time (s)")
plt.ylabel("Absolute Error (degrees)")
plt.title("Prediction Errors Over Time")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "errors_comparison.png"),
    dpi=300,
)

plt.close()


# Plot 3: ROM comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["modality"],
    results_df["rom_deg"],
)

plt.ylabel("Range of Motion (degrees)")
plt.title("ROM Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "rom_comparison.png"),
    dpi=300,
)

plt.close()


print(f"\nResults saved to: {output_dir}")
