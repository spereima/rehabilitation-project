
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.models import get_model
from src.data.sils_folder_dataset import SilsFolderDataset
from src.data.rgb_folder_dataset import RGBFolderDataset


@hydra.main(
    config_path="../../conf",
    config_name="default",
    version_base="1.3"
)
def main(cfg: DictConfig):

    modality = cfg.data.modality

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Модель:", modality)
    print("Device:", device)

    run_dir = (
        Path("output")
        / cfg.project.name
        / modality
        / cfg.data.split_name
        / cfg.project.run_name
    )

    checkpoint_path = (
        run_dir / "checkpoints" / "epoch_3.pth"
    )

    manifest_dir = (
        Path("data/manifests")
        / cfg.project.name
        / modality
        / cfg.data.split_name
        / cfg.project.run_name
    )

    norm_path = (
        manifest_dir
        / f"{cfg.data.split_name}_norm_params.json"
    )

    assert checkpoint_path.exists(), checkpoint_path
    assert norm_path.exists(), norm_path

    with open(norm_path) as file:
        norm_params = json.load(file)

    mean = float(norm_params["mean"])
    std = float(norm_params["std"])

    assert std > 0

    DatasetClass = (
        SilsFolderDataset
        if cfg.data.dataset_loader == "SilsFolderDataset"
        else RGBFolderDataset
    )

    dataset = DatasetClass(cfg, active_split="test")

    print("Тестових кадрів:", len(dataset))
    print("Нормалізація:", norm_params)

    assert len(dataset) == 223335

    ModelClass = get_model(cfg.model)
    model = ModelClass(cfg.model).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    assert checkpoint["epoch"] == 3

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results_dir = (
        Path("output")
        / cfg.project.name
        / "test_results"
        / modality
    )

    parts_dir = results_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg.training.batch_size
    batches_per_part = 100
    frames_per_part = batch_size * batches_per_part

    total_parts = (
        len(dataset) + frames_per_part - 1
    ) // frames_per_part

    for part_number in tqdm(
        range(total_parts),
        desc=f"{modality} — тестування",
        unit="частина"
    ):

        part_path = (
            parts_dir / f"part_{part_number:04d}.csv"
        )

        start = part_number * frames_per_part
        end = min(
            start + frames_per_part,
            len(dataset)
        )

        if part_path.exists():
            existing = pd.read_csv(part_path)

            if len(existing) == end - start:
                continue

        subset = Subset(
            dataset,
            range(start, end)
        )

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=device.type == "cuda"
        )

        predictions = []

        with torch.inference_mode():

            for x, y_norm, meta in loader:

                x = x.to(device)

                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda"
                ):
                    y_pred_norm = model(x)

                y_pred = (
                    y_pred_norm.float().reshape(-1)
                    * std + mean
                )

                predictions.extend(
                    y_pred.cpu().tolist()
                )

        part_df = dataset.df.iloc[start:end].copy()

        assert len(part_df) == len(predictions)

        part_df["predicted_angle"] = predictions

        part_df["ground_truth_angle"] = (
            part_df["angle_deg"].astype(float)
        )

        part_df["absolute_error"] = (
            part_df["predicted_angle"]
            - part_df["ground_truth_angle"]
        ).abs()

        part_df["squared_error"] = (
            part_df["predicted_angle"]
            - part_df["ground_truth_angle"]
        ) ** 2

        part_df = part_df[
            [
                "subject",
                "exercise",
                "camera",
                "frame_id",
                "img_path",
                "ground_truth_angle",
                "predicted_angle",
                "absolute_error",
                "squared_error"
            ]
        ]

        temporary_path = part_path.with_suffix(".tmp")

        part_df.to_csv(
            temporary_path,
            index=False
        )

        temporary_path.replace(part_path)

    # Об'єднуємо результати
    all_parts = []

    for part_number in range(total_parts):
        part_path = (
            parts_dir / f"part_{part_number:04d}.csv"
        )

        assert part_path.exists(), part_path

        all_parts.append(
            pd.read_csv(
                part_path,
                dtype={"subject": str}
            )
        )

    results = pd.concat(
        all_parts,
        ignore_index=True
    )

    assert len(results) == len(dataset)

    output_path = results_dir / "all_predictions.csv"

    results.to_csv(
        output_path,
        index=False
    )

    mae = results["absolute_error"].mean()

    rmse = np.sqrt(
        results["squared_error"].mean()
    )

    summary = {
        "modality": modality,
        "test_frames": len(results),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "checkpoint_epoch": 3
    }

    with open(results_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    print("\nТЕСТУВАННЯ ЗАВЕРШЕНО")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("Результати:", output_path)


if __name__ == "__main__":
    main()
