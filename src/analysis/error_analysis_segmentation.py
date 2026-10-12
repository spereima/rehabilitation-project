
import json
import os
import shutil

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

if sequence.empty:
    raise ValueError("No predictions found for this sequence.")

sequence["absolute_error"] = abs(
    sequence["angle_pred"] - sequence["angle_gt"]
)

largest_errors = sequence.sort_values(
    "absolute_error",
    ascending=False,
).head(10)

output_dir = (
    f"results/{run_name}/largest_errors"
)

os.makedirs(output_dir, exist_ok=True)

largest_errors.to_csv(
    f"results/{run_name}/largest_errors.csv",
    index=False,
)

print(
    largest_errors[
        [
            "frame_id",
            "angle_gt",
            "angle_pred",
            "absolute_error",
            "img_path",
        ]
    ].to_string(index=False)
)

for _, row in largest_errors.iterrows():
    source_path = row["img_path"]

    filename = (
        f"frame_{int(row['frame_id']):04d}_"
        f"error_{row['absolute_error']:.2f}.png"
    )

    destination_path = os.path.join(
        output_dir,
        filename,
    )

    if os.path.exists(source_path):
        shutil.copy(
            source_path,
            destination_path,
        )
    else:
        print(f"Image not found: {source_path}")
