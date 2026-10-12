
import json

import pandas as pd
from scipy.signal import find_peaks

import os

import matplotlib.pyplot as plt




run_name = "semantic_segmentation_split1_01_05_cam0"

predictions_path = (
    "output/ucophyrehapp_sample/semantic_segmentation/split_1/"
    f"{run_name}/train_predictions.csv"
)

norm_params_path = (
    "data/manifests/ucophyrehapp_sample/"
    "semantic_segmentation/split_1/"
    f"{run_name}/split_1_norm_params.json"
)

predictions = pd.read_csv(predictions_path)

predictions["subject"] = (
    predictions["subject"]
    .astype(str)
    .str.replace("tensor(", "", regex=False)
    .str.replace(")", "", regex=False)
    .astype(int)
)

predictions["exercise"] = (
    predictions["exercise"]
    .astype(str)
    .str.replace("tensor(", "", regex=False)
    .str.replace(")", "", regex=False)
    .astype(int)
)

with open(norm_params_path, "r") as file:
    norm_params = json.load(file)

mean = norm_params["mean"]
std = norm_params["std"]

predictions["angle_gt"] = (
    predictions["angle_norm_gt"] * std + mean
)

exercise = 1
subject = 0
camera = "cam0"

sequence = predictions[
    (predictions["exercise"] == exercise)
    & (predictions["subject"] == subject)
    & (predictions["camera"] == camera)
].copy()

sequence = sequence.sort_values("frame_id").reset_index(drop=True)

if sequence.empty:
    raise ValueError("No predictions found for this sequence.")

angle = sequence["angle_gt"]

peaks, _ = find_peaks(
    -angle,
    distance=100,
    prominence=20,
)

print("Detected minima:")

for peak in peaks:
    print(
        f"Frame {int(sequence.iloc[peak]['frame_id'])}: "
        f"{angle.iloc[peak]:.2f} degrees"
    )


fps = 30
delta_t = 1 / fps

sequence["time"] = (
    sequence["frame_id"] - sequence["frame_id"].iloc[0]
) / fps


rom = angle.max() - angle.min()

sequence["angular_velocity"] = (
    angle.diff() / delta_t
)

mean_angular_velocity = (
    sequence["angular_velocity"].abs().mean()
)

peak_angular_velocity = (
    sequence["angular_velocity"].abs().max()
)

sequence["velocity_change"] = (
    sequence["angular_velocity"].diff().abs()
)

smoothness = sequence["velocity_change"].mean()


repetition_boundaries = [
    (5, 164),
    (164, 320),
    (320, 462),
    (462, 624),
]

repetitions = []

for i in range(len(repetition_boundaries)):
    start_frame = repetition_boundaries[i][0]
    end_frame = repetition_boundaries[i][1]

    repetition = sequence[
        (sequence["frame_id"] >= start_frame)
        & (sequence["frame_id"] <= end_frame)
    ].copy()

    repetition_angle = repetition["angle_gt"]

    repetition_rom = (
        repetition_angle.max()
        - repetition_angle.min()
    )

    repetition_velocity = (
        repetition_angle.diff() / delta_t
    )

    repetition_peak_velocity = (
        repetition_velocity.abs().max()
    )

    repetition_smoothness = (
        repetition_velocity.diff().abs().mean()
    )

    repetition_duration = (
        repetition["time"].iloc[-1]
        - repetition["time"].iloc[0]
    )

    repetitions.append(
        {
            "repetition": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_s": repetition_duration,
            "rom_deg": repetition_rom,
            "peak_velocity_deg_s": repetition_peak_velocity,
            "smoothness_proxy": repetition_smoothness,
        }
    )

repetitions_df = pd.DataFrame(repetitions)

print()
print("Repetition descriptors:")
print(repetitions_df.to_string(index=False))

print()
print(f"Exercise: {exercise:02d}")
print(f"Subject: {subject}")
print(f"Camera: {camera}")
print(f"Frames: {len(sequence)}")
print(f"FPS: {fps}")

print(f"ROM: {rom:.2f} degrees")
print(
    f"Mean angular velocity: "
    f"{mean_angular_velocity:.2f} degrees/s"
)
print(
    f"Peak angular velocity: "
    f"{peak_angular_velocity:.2f} degrees/s"
)
print(
    f"Smoothness proxy: "
    f"{smoothness:.2f} degrees/s"
)

output_dir = (
    "results/semantic_segmentation_split1_01_05_cam0/"
    "movement_descriptors"
)

os.makedirs(output_dir, exist_ok=True)

repetitions_df.to_csv(
    os.path.join(output_dir, "repetition_descriptors.csv"),
    index=False,
)

first_last = pd.concat(
    [
        repetitions_df.iloc[[0]],
        repetitions_df.iloc[[-1]],
    ]
)

first_last.to_csv(
    os.path.join(output_dir, "first_vs_last_repetition.csv"),
    index=False,
)

print()
print("First vs last repetition:")
print(first_last.to_string(index=False))

descriptors = pd.DataFrame(
    [
        {
            "exercise": exercise,
            "subject": subject,
            "camera": camera,
            "frames": len(sequence),
            "fps": fps,
            "rom_deg": rom,
            "mean_velocity_deg_s": mean_angular_velocity,
            "peak_velocity_deg_s": peak_angular_velocity,
            "smoothness_proxy": smoothness,
        }
    ]
)

descriptors.to_csv(
    os.path.join(output_dir, "movement_descriptors.csv"),
    index=False,
)

# Ground Truth Angle Over Time
plt.figure(figsize=(12, 6))

plt.plot(
    sequence["time"],
    sequence["angle_gt"],
    label="Ground Truth Angle",
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title(
    f"Ground Truth Angle Over Time | "
    f"Exercise {exercise:02d}, {camera}, Subject {subject}"
)

plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "angle_over_time.png"),
    dpi=300,
)

plt.close()

# Angular Velocity Over Time
plt.figure(figsize=(12, 6))

plt.plot(
    sequence["time"],
    sequence["angular_velocity"],
    label="Angular Velocity",
)

plt.axhline(y=0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (degrees/s)")
plt.title(
    f"Angular Velocity Over Time | "
    f"Exercise {exercise:02d}, {camera}, Subject {subject}"
)

plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "angular_velocity_over_time.png"),
    dpi=300,
)

plt.close()
