import os

import matplotlib.pyplot as plt
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

sequence = sequence.sort_values("frame_id")

sequence["absolute_error"] = abs(
    sequence["angle_pred"] - sequence["angle_gt"]
)

sequence_mae = sequence["absolute_error"].mean()

os.makedirs(
    "results/sils_split1_01_05_cam0",
    exist_ok=True,
)

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
    "results/sils_split1_01_05_cam0/"
    "ground_truth_vs_predicted_ex01.png",
    dpi=300,
)

plt.show()

print(f"Number of frames: {len(sequence)}")
print(f"Sequence MAE: {sequence_mae:.4f} degrees")