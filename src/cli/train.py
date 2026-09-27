# train.py
import os
import sys
import pandas as pd
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm
from tqdm import tqdm

from src.utils.common import set_all_seeds
from src.models import get_model
from src.loss import get_loss
from src.metric import get_metric
from src.data.sils_folder_dataset import SilsFolderDataset  # dataset que carga desde CSVs
from src.data.rgb_folder_dataset import RGBFolderDataset
from src.logging.tensorboard_logger import TBLogger


# ───────────────────────────── Métricas simples ─────────────────────────────
def mae(y, yhat):
    return torch.mean(torch.abs(yhat - y))

def rmse(y, yhat):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


# ───────────────────────────── Entrenamiento ─────────────────────────────
@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # Build logs folder and create train_log.txt file
    train_log_path = os.path.join("output", cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name, "logs")
    train_log_file = os.path.join(train_log_path, "train_log.txt")
    os.makedirs(train_log_path, exist_ok=True)
    # Open train_log.txt file
    sys.stdout = open(train_log_file, "w")

    # Print config on screen and log file
    print("CONFIG:\n", OmegaConf.to_yaml(cfg))
    
    set_all_seeds(cfg.training.seed)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # Create Tensorboard object
    if cfg.logging.tensorboard_log:
        tb_logger = TBLogger(cfg)
    else:
        tb_logger = None

    # Dataset de entrenamiento (ya normalizado)
    manifests_dir = 'data/manifests'
    csv_path = os.path.join(manifests_dir, cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name)
    train_csv = os.path.join(csv_path, f"{cfg.data.split_name}_train.csv")
    assert os.path.exists(train_csv), f"Manifest CSV no encontrado: {train_csv}"

    # Cargar datos de desnormalización
    denorm_params_path = os.path.join(csv_path, f"{cfg.data.split_name}_norm_params.json")
    assert os.path.exists(denorm_params_path), f"Parámetros de normalización no encontrados: {denorm_params_path}"
    norm_params = {}
    with open(denorm_params_path, "r") as f:
        norm_params = json.load(f)

    # TODO - Cambiar al que venga de config
    # train_ds = RGBFolderDataset(cfg, active_split="train")
    DatasetClass = SilsFolderDataset if cfg.data.dataset_loader == "SilsFolderDataset" else RGBFolderDataset
    train_ds = DatasetClass(cfg, active_split="train")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Modelo y optimizador
    model = get_model(cfg.model)
    model = model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.training.amp)

    # Loss function
    loss_fn = get_loss(cfg.loss)

    # Loop de entrenamiento
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        running = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")

        for i, (x, y_norm, meta) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y_norm.to(device).float()   # target ya normalizado

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_str, enabled=cfg.training.amp):
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

            scaler.scale(loss).backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(opt)
            scaler.update()

            # Desnormalizar y calcular las métricas
            with torch.no_grad():
                y_denorm = y * norm_params["std"] + norm_params["mean"]
                y_pred_denorm = y_pred * norm_params["std"] + norm_params["mean"]

                mae_v = mae(y_denorm, y_pred_denorm)
                rmse_v = rmse(y_denorm, y_pred_denorm)

            running["loss"]  += float(loss)
            running["mae"]   += float(mae_v)
            running["rmse"]  += float(rmse_v)

            # Logging screen and log file
            if i % cfg.logging.print_every == 0:
                pbar.set_postfix({
                    "loss": f"{running['loss']/i:.4f}",
                    "MAE": f"{running['mae']/i:.3f}",
                    "RMSE": f"{running['rmse']/i:.3f}",
                })
        
        # Print train epoch results
        n_batches = len(train_loader)
        print(f"[train] Epoch {epoch}: Loss={running['loss']/n_batches:.4f}, "
              f"MAE={running['mae']/n_batches:.3f}, RMSE={running['rmse']/n_batches:.3f}")
        
        # Log to tensorboard
        if tb_logger is not None:
            metrics = {
                "MAE": running['mae']/n_batches,
                "RMSE": running['rmse']/n_batches
            }
            tb_logger.log_epoch(cfg, epoch, running['loss']/n_batches, metrics)
            
        # Checkpoint saver
        if epoch % cfg.training.checkpoint_save_every == 0 or epoch == cfg.training.epochs:
            ckpt_path = os.path.join("output", cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name, "checkpoints")
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_path = os.path.join(ckpt_path, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt_path)
            print(f"[save] checkpoint -> {ckpt_path}")

        # Validation
        if cfg.validation.enabled and (epoch % cfg.validation.every_n_epochs == 0 or epoch == cfg.training.epochs):
            # val_ds = RGBFolderDataset(cfg, active_split="val") # TODO - Cambiar al que venga de config
            val_ds = DatasetClass(cfg, active_split="val")
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=False
            )

            model.eval()
            val_running = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
            with torch.no_grad():
                for x_val, y_val_norm, meta_val in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                    x_val = x_val.to(device)
                    y_val = y_val_norm.to(device).float()

                    with torch.amp.autocast(device_type=device_str, enabled=cfg.training.amp):
                        y_val_pred = model(x_val)
                        val_loss = F.mse_loss(y_val_pred, y_val)

                    # Desnormalizar y calcular las métricas
                    y_val_denorm = y_val * norm_params["std"] + norm_params["mean"]
                    y_val_pred_denorm = y_val_pred * norm_params["std"] + norm_params["mean"]

                    val_mae_v = mae(y_val_denorm, y_val_pred_denorm)
                    val_rmse_v = rmse(y_val_denorm, y_val_pred_denorm)

                    val_running["loss"] += float(val_loss)
                    val_running["mae"] += float(val_mae_v)
                    val_running["rmse"] += float(val_rmse_v)

            n_val_batches = len(val_loader)
            print(f"[validation] Epoch {epoch}: Loss={val_running['loss']/n_val_batches:.4f}, "
                  f"MAE={val_running['mae']/n_val_batches:.3f}, RMSE={val_running['rmse']/n_val_batches:.3f}")

    print("Entrenamiento finalizado.")

    # Guardar predicciones del entrenamiento
    preds = []
    model.eval()
    with torch.no_grad():
        for x, y, meta in DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False,
                                     num_workers=cfg.training.num_workers, pin_memory=True):
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            y_gt = y.cpu().numpy()

            for j in range(x.size(0)):
                preds.append({
                    "subject": meta["subject"][j],
                    "exercise": meta["exercise"][j],
                    "camera": meta["camera"][j],
                    "frame_id": int(meta["frame_id"][j]),
                    "img_path": meta["img_path"][j],
                    "angle_norm_gt": float(y_gt[j]),
                    "angle_norm_pred": float(y_pred[j]),
                })

    preds_path = os.path.join("output", cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name)
    os.makedirs(preds_path, exist_ok=True)
    pd.DataFrame(preds).to_csv(os.path.join(preds_path, 'train_predictions.csv'), index=True)
    print(f"[save] predictions -> {preds_path}")

    # Close log file
    sys.stdout.close()

if __name__ == "__main__":
    main()
