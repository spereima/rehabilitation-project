
"""Plot movement descriptors for the final test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(
    "results/final_test/movement_descriptors/movement_descriptors.csv"
)

OUTPUT_PATH = Path(
    "results/final_test/movement_descriptors/"
    "movement_descriptors_comparison.png"
)

data = pd.read_csv(DATA_PATH)

metrics = [
    ("rom_deg", "Range of Motion", "degrees"),
    ("mean_velocity_deg_s", "Mean Angular Velocity", "deg/s"),
    ("peak_velocity_deg_s", "Peak Angular Velocity", "deg/s"),
    ("smoothness_proxy", "Smoothness Proxy", "deg/s²")
]

modalities = ["sils", "semantic_segmentation"]

labels = ["SILS", "Semantic Segmentation"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for index, (metric, title, unit) in enumerate(metrics):
    ax = axes[index // 2][index % 2]

    gt_values = []
    pred_values = []

    for modality in modalities:
        subset = data[data["modality"] == modality]

        gt_values.append(
            subset["gt_" + metric].mean()
        )

        pred_values.append(
            subset["pred_" + metric].mean()
        )

    positions = np.arange(len(modalities))
    width = 0.35

    ax.bar(
        positions - width / 2,
        gt_values,
        width,
        label="Ground Truth"
    )

    ax.bar(
        positions + width / 2,
        pred_values,
        width,
        label="Predicted"
    )

    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "Movement Descriptors: Ground Truth vs. Predicted",
    fontsize=16
)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", OUTPUT_PATH)
