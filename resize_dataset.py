from pathlib import Path
from PIL import Image
from tqdm import tqdm

source_folder = Path("dataset_check_frames")
target_folder = Path("dataset_check_frames_small")

for representation in ["sils", "semantic_segmentation"]:
    source_path = source_folder / representation
    target_path = target_folder / representation

    files = sorted(source_path.rglob("*.png"))
    
    for file_path in tqdm(files):
        relative_path = file_path.relative_to(source_path)
        output_path = target_path / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        with Image.open(file_path) as image:
            resized_image = image.resize(
                (224, 224),
                resample=Image.Resampling.NEAREST
            )
            resized_image.save(output_path, format="PNG")