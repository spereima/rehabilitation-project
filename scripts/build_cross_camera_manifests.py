import os
import json
import pandas as pd

modalities = ["sils", "semantic_segmentation"]
cameras = ["cam1", "cam2", "cam3", "cam4"]

split_name = "split_2"
run_name = "full_16_exercises_cam0"

with open("data/splits/split_2_27_subjects.json", "r") as file:
    test_subjects = json.load(file)["test"]

with open("dataset_check/dataset_3d_with_angles.json", "r") as file:
    metadata = json.load(file)

labels = {}

for sequence in metadata.get("data", metadata):
    subject = str(sequence.get("folder"))
    exercise = str(sequence.get("exercise"))

    for frame in sequence.get("frames") or []:
        frame_id = frame.get("id")
        angle = frame.get("joints", {}).get("angle")

        if frame_id is None or angle is None or angle == 0:
            continue

        labels[(subject, exercise, int(frame_id))] = float(angle)

for modality in modalities:
    original_dir = os.path.join(
        "data/manifests/project_name",
        modality,
        split_name,
        run_name
    )

    with open(
        os.path.join(original_dir, "split_2_norm_params.json"), "r"
    ) as file:
        norm_params = json.load(file)

    mean = norm_params["mean"]
    std = norm_params["std"]

    for camera in cameras:
        rows = []

        for subject in test_subjects:
            for exercise_number in range(1, 17):
                exercise = f"{exercise_number:02d}"

                folder = os.path.join(
                    "dataset_check_frames",
                    modality,
                    subject,
                    exercise,
                    camera
                )

                if not os.path.isdir(folder):
                    continue

                for filename in sorted(os.listdir(folder)):
                    if not filename.endswith(".png"):
                        continue

                    frame_id = int(os.path.splitext(filename)[0])
                    angle = labels.get(
                        (subject, exercise, frame_id)
                    )

                    if angle is None:
                        continue

                    image_path = os.path.join(folder, filename)

                    rows.append({
                        "subject": subject,
                        "exercise": exercise,
                        "camera": camera,
                        "frame_id": frame_id,
                        "img_path": image_path,
                        "angle_deg": angle,
                        "angle_norm": (angle - mean) / std
                    })

        output_dir = os.path.join(
            "data/manifests/project_name",
            modality,
            split_name,
            f"test_{camera}"
        )

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir, "split_2_test.csv"
        )

        pd.DataFrame(rows).to_csv(output_path, index=False)

        print(
            f"{modality} | {camera}: "
            f"{len(rows)} labeled frames"
        )

print("Cross-camera manifests completed!")