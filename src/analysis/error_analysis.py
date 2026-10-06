import os
import shutil

import pandas as pd


predictions_path = (
    "output/ucophyrehapp_sample/sils/split_1/"
    "sils_split1_01_05_cam0/test_predictions.csv"
)

predictions = pd.read_csv(predictions_path)

exercise = 1
subject = 1
camera = "cam0"

sequence = predictions[
    (predictions["exercise"] == exercise)
    & (predictions["subject"] == subject)
    & (predictions["camera"] == camera)
].copy()

sequence["absolute_error"] = abs(
    sequence["angle_pred"] - sequence["angle_gt"]
)

largest_errors = sequence.sort_values(
    "absolute_error",
    ascending=False,
).head(10)

output_dir = (
    "results/sils_split1_01_05_cam0/"
    "largest_errors"
)

os.makedirs(output_dir, exist_ok=True)

largest_errors.to_csv(
    "results/sils_split1_01_05_cam0/"
    "largest_errors.csv",
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