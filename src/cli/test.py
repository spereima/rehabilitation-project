# train.py
import os
import sys
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm
from tqdm import tqdm
import pandas as pd

from src.utils.common import set_all_seeds
from src.models import get_model
from src.loss import get_loss
from src.data.rgb_folder_dataset import RGBFolderDataset  # dataset que carga desde CSVs
from src.data.sils_folder_dataset import SilsFolderDataset


# ───────────────────────────── Métricas simples ─────────────────────────────
def mae(y, yhat):
    return torch.mean(torch.abs(yhat - y))

def rmse(y, yhat):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # Build logs folder and create test_log.txt file
    log_path = os.path.join("output", cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name, "logs")
    test_log_file = os.path.join(log_path, "test_log.txt")
    os.makedirs(log_path, exist_ok=True)
    # Open test_log.txt file
    sys.stdout = open(test_log_file, "w")
    
    
    print("CONFIG:\n", OmegaConf.to_yaml(cfg))
    set_all_seeds(cfg.training.seed)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # Dataset de entrenamiento (ya normalizado)
    manifests_dir = 'data/manifests'
    csv_path = os.path.join(manifests_dir, cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name)
    test_csv = os.path.join(csv_path, f"{cfg.data.split_name}_test.csv")
    assert os.path.exists(test_csv), f"Manifest CSV no encontrado: {test_csv}"
    
    # test_ds = RGBFolderDataset(cfg, active_split="test")
    DatasetClass = SilsFolderDataset if cfg.data.dataset_loader == "SilsFolderDataset" else RGBFolderDataset
    test_ds = DatasetClass(cfg, active_split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset de test cargado: {len(test_ds)} muestras.")

    # Cargar datos de desnormalización
    denorm_params_path = os.path.join(csv_path, f"{cfg.data.split_name}_norm_params.json")
    assert os.path.exists(denorm_params_path), f"Parámetros de normalización no encontrados: {denorm_params_path}"
    norm_params = {}
    with open(denorm_params_path, "r") as f:
        norm_params = json.load(f)

    # Cargar modelo desde checkpoint
    model = get_model(cfg.model)
    epoch = cfg.test.checkpoint_epoch if cfg.test.checkpoint_epoch is not None else cfg.training.epochs
    ckpt_path = os.path.join("output", cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name, "checkpoints", f"epoch_{epoch}.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint no encontrado: {ckpt_path}"

    # ckpt = torch.load(ckpt_path, map_location=device)
    model = model(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    model.eval()
    print(f"Model checkpoint epoch {epoch}.")

    # Evaluación en test
    all_preds = []
    all_targets = []
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluando en test")
        for i, (x, y_norm, meta) in enumerate(pbar, start=1):

            x = x.to(device)  # [B,1,H,W]
            y = y_norm.to(device)  # [B]

            with torch.amp.autocast(device_type=device_str, enabled=cfg.training.amp):
                y_pred = model(x)  # [B]

            y_denorm = y * norm_params["std"] + norm_params["mean"]
            y_pred_denorm = y_pred * norm_params["std"] + norm_params["mean"]
            
            for j in range(x.size(0)):
                predictions.append({
                    "subject": int(meta["subject"][j]),
                    "exercise": int(meta["exercise"][j]),
                    "camera": meta["camera"][j],
                    "frame_id": int(meta["frame_id"][j]),
                    "img_path": meta["img_path"][j],
                    "angle_gt": float(y_denorm[j]),
                    "angle_pred": float(y_pred_denorm[j]),
                })

            all_preds.append(y_pred_denorm.cpu())
            all_targets.append(y_denorm.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    test_mae = mae(all_targets, all_preds).item()
    test_rmse = rmse(all_targets, all_preds).item()

    print(f"Test results - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    predictions_path = os.path.join(
        "output",
        cfg.project.name,
        cfg.data.modality,
        cfg.data.split_name,
        cfg.project.run_name,
        "test_predictions.csv",
    )

    pd.DataFrame(predictions).to_csv(
        predictions_path,
        index=False,
    )

    print(f"Test predictions saved to {predictions_path}")
    

if __name__ == "__main__":
    main()