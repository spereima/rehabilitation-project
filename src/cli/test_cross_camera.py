
import os
import json

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    camera = cfg.test_camera

    model_run = "full_16_exercises_cam0"
    test_run = f"test_{camera}"

    base_dir = os.path.join(
        "data/manifests",
        cfg.project.name,
        modality,
        cfg.data.split_name
    )

    test_dir = os.path.join(base_dir, test_run)
    model_dir = os.path.join(base_dir, model_run)

    test_csv = os.path.join(
        test_dir,
        f"{cfg.data.split_name}_test.csv"
    )

    norm_path = os.path.join(
        model_dir,
        f"{cfg.data.split_name}_norm_params.json"
    )

    checkpoint_path = os.path.join(
        "output",
        cfg.project.name,
        modality,
        cfg.data.split_name,
        model_run,
        "checkpoints",
        "best.pth"
    )

    output_dir = os.path.join(
        "output",
        cfg.project.name,
        modality,
        cfg.data.split_name,
        model_run,
        "cross_camera_tests",
        camera
    )

    os.makedirs(output_dir, exist_ok=True)

    for path in [test_csv, norm_path, checkpoint_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(norm_path, "r") as file:
        norm_params = json.load(file)

    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Modality: {modality}")
    print(f"Camera: {camera}")
    print(f"Device: {device}")

    # Dataset classes use cfg.project.run_name
    # to find the test manifest.
    from omegaconf import OmegaConf

    test_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True)
    )
    test_cfg.project.run_name = test_run

    if modality == "sils":
        DatasetClass = SilsFolderDataset
    else:
        DatasetClass = RGBFolderDataset

    test_dataset = DatasetClass(
        test_cfg,
        active_split="test"
    )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    ModelClass = get_model(cfg.model)
    model = ModelClass(cfg.model).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test frames: {len(test_dataset)}")

    predictions = []

    with torch.no_grad():
        for x, y_norm, meta in tqdm(test_loader):
            x = x.to(device)

            predicted_norm = model(x).float().reshape(-1)
            actual_norm = y_norm.float().reshape(-1)

            predicted = (
                predicted_norm.cpu() * norm_params["std"]
                + norm_params["mean"]
            )

            actual = (
                actual_norm * norm_params["std"]
                + norm_params["mean"]
            )

            for i in range(len(predicted)):
                predictions.append({
                    "subject": int(meta["subject"][i]),
                    "exercise": int(meta["exercise"][i]),
                    "camera": meta["camera"][i],
                    "frame_id": int(meta["frame_id"][i]),
                    "img_path": meta["img_path"][i],
                    "angle_gt": float(actual[i]),
                    "angle_pred": float(predicted[i])
                })

    results = pd.DataFrame(predictions)

    error = results["angle_pred"] - results["angle_gt"]

    test_mae = error.abs().mean()
    test_rmse = (error.pow(2).mean()) ** 0.5

    predictions_path = os.path.join(
        output_dir, "test_predictions.csv"
    )

    results.to_csv(predictions_path, index=False)

    metrics = {
        "modality": modality,
        "camera": camera,
        "test_frames": len(results),
        "mae": float(test_mae),
        "rmse": float(test_rmse)
    }

    with open(
        os.path.join(output_dir, "metrics.json"), "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
