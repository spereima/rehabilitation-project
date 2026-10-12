
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


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
    .str.replace("tensor(", "", regex=False)
    .str.replace(")", "", regex=False)
    .astype(int)
)

predictions["exercise"] = (
    predictions["exercise"]
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

predictions["angle_pred"] = (
    predictions["angle_norm_pred"] * std + mean
)

exercise = 1
subject = 0
camera = "cam0"

sequence = predictions[
    (predictions["exercise"] == exercise)
    & (predictions["subject"] == subject)
    & (predictions["camera"] == camera)
].copy()

sequence = sequence.sort_values("frame_id")

if sequence.empty:
    raise ValueError("No predictions found for this sequence.")

sequence["absolute_error"] = abs(
    sequence["angle_pred"] - sequence["angle_gt"]
)

sequence_mae = sequence["absolute_error"].mean()

output_dir = f"results/{run_name}"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(
    sequence["frame_id"],
    sequence["angle_gt"],
    label="Ground Truth",
)

plt.plot(
    sequence["frame_id"],
    sequence["angle_pred"],
    label="Predicted",
)

plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")

plt.title(
    f"Exercise {exercise:02d}, {camera}, "
    f"Subject {subject} | MAE = {sequence_mae:.2f}°"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        f"ground_truth_vs_predicted_ex{exercise:02d}.png",
    ),
    dpi=300,
)

plt.show()

print(f"Number of frames: {len(sequence)}")
print(f"Sequence MAE: {sequence_mae:.4f} degrees")
