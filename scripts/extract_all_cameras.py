
import json
from pathlib import Path

import cv2
from tqdm import tqdm


SOURCE_ROOT = Path("dataset_check")
TARGET_ROOT = Path("dataset_check_frames")

MODALITIES = ["sils", "semantic_segmentation"]
CAMERAS = ["cam1", "cam2", "cam3", "cam4"]

subjects = sorted(
    [
        folder.name
        for folder in (SOURCE_ROOT / "sils").iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ],
    key=int
)

assert len(subjects) == 27, (
    f"Participants found: {len(subjects)}"
)

print("Subjects:", subjects)
print("Total subjects:", len(subjects))

missing_videos = []
failed_videos = []
total_new_frames = 0

for modality in MODALITIES:

    print(f"\nProcessing {modality}")

    for subject in tqdm(subjects, desc=modality):

        for exercise_number in range(1, 17):

            exercise = f"{exercise_number:02d}"

            for camera in CAMERAS:

                video_path = (
                    SOURCE_ROOT
                    / modality
                    / subject
                    / exercise
                    / f"{camera}.mp4"
                )

                output_folder = (
                    TARGET_ROOT
                    / modality
                    / subject
                    / exercise
                    / camera
                )

                if not video_path.is_file():
                    missing_videos.append(str(video_path))
                    continue

                video = cv2.VideoCapture(str(video_path))

                if not video.isOpened():
                    failed_videos.append(str(video_path))
                    video.release()
                    continue

                output_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                frame_number = 0

                while True:

                    success, frame = video.read()

                    if not success:
                        break

                    frame_path = (
                        output_folder
                        / f"{frame_number:04d}.png"
                    )

                    if not frame_path.exists():

                        saved = cv2.imwrite(
                            str(frame_path),
                            frame
                        )

                        if not saved:
                            raise RuntimeError(
                                f"Cannot save: {frame_path}"
                            )

                        total_new_frames += 1

                    frame_number += 1

                video.release()

print("\n" + "=" * 50)
print("EXTRACTION COMPLETED")
print("=" * 50)

print("New frames:", total_new_frames)
print("Missing videos:", len(missing_videos))
print("Failed videos:", len(failed_videos))

if missing_videos:
    print("\nFirst missing videos:")
    for path in missing_videos[:20]:
        print(path)

if failed_videos:
    print("\nFirst failed videos:")
    for path in failed_videos[:20]:
        print(path)
