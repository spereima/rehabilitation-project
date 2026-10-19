import os
import json
import cv2

modalities = ["sils", "semantic_segmentation"]
cameras = ["cam1", "cam2", "cam3", "cam4"]

with open("data/splits/split_2_27_subjects.json", "r") as file:
    splits = json.load(file)

test_subjects = splits["test"]

for modality in modalities:
    for subject in test_subjects:
        for exercise_number in range(1, 17):
            exercise = f"{exercise_number:02d}"

            for camera in cameras:
                video_path = os.path.join(
                    "dataset_check",
                    modality,
                    subject,
                    exercise,
                    f"{camera}.mp4"
                )

                output_folder = os.path.join(
                    "dataset_check_frames",
                    modality,
                    subject,
                    exercise,
                    camera
                )

                if not os.path.exists(video_path):
                    print(f"Missing video: {video_path}")
                    continue

                os.makedirs(output_folder, exist_ok=True)

                video = cv2.VideoCapture(video_path)

                if not video.isOpened():
                    print(f"Cannot open: {video_path}")
                    continue

                frame_number = 0

                while True:
                    success, frame = video.read()

                    if not success:
                        break

                    frame_name = f"{frame_number:04d}.png"
                    frame_path = os.path.join(
                        output_folder, frame_name
                    )

                    if not os.path.exists(frame_path):
                        cv2.imwrite(frame_path, frame)

                    frame_number += 1

                video.release()

                print(
                    f"{modality} | subject {subject} | "
                    f"exercise {exercise} | {camera}: "
                    f"{frame_number} frames"
                )

print("Extraction completed!")