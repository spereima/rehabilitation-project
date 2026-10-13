import os, json, re, glob, csv
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from src.normalization import get_normalization_function

def map_subject(s: str, cfg) -> str:
    # Esta función se usa para mapear los nombres de los folders, por si son diferentes a lo que viene en el JSON
    t = cfg.data.subject_to_folder.type
    if t == "identity":
        return s
    if t == "strip_S":
        return s.lstrip("S").lstrip("s")
    if t == "int":
        # convierte "S07" -> "7"
        return str(int(re.findall(r"\d+", s)[0]))
    return s  # fallback

@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print("CONFIG:\n", OmegaConf.to_yaml(cfg))

    # 1) Cargar SPLIT
    split_path = cfg.data.split_path
    with open(split_path, "r") as f:
        split = json.load(f)

    splits = cfg.data.splits

    for active_split in splits: # Recorremos todos los splits y construimos todo el manifest normalizado
        allowed_subjects = set(split[active_split])
        print(f"[split] {active_split}: {sorted(allowed_subjects)}")

        # --- tras cargar el split ---
        split_subjects = set(split[active_split])

        # Filtros opcionales (si listas vacías, no restringen)
        inc_subj = set(cfg.data.include_subjects) if cfg.data.include_subjects else None
        inc_ex   = set(cfg.data.include_exercises) if cfg.data.include_exercises else None
        inc_cam  = set(cfg.data.include_cameras) if cfg.data.include_cameras else None

        # Sujetos permitidos = intersección entre split y filtro opcional (si lo hay)
        # if inc_subj is not None:
        #     allowed_subjects = inc_subj
        # else:
        #     allowed_subjects = split_subjects
        if inc_subj is not None:
            allowed_subjects = split_subjects & inc_subj
        else:
            allowed_subjects = split_subjects

        print(f"[filters] subjects={sorted(list(allowed_subjects))}")
        print(f"[filters] exercises={sorted(list(inc_ex)) if inc_ex else 'ALL'}  "
            f"cameras={sorted(list(inc_cam)) if inc_cam else 'ALL'}")

        # 2) Cargar METADATA JSON e indexar ángulos válidos
        with open(cfg.data.metadata_json, "r") as f:
            meta = json.load(f)

        labels = {}  # (folder, exercise, frame_id_int) -> angle_deg
        data_blocks = meta.get("data", meta)  # por si viene “plano” o bajo "data"
        kept, skipped = 0, 0
        for seq in data_blocks:
            # Si no hay frames válidos, skip
            if seq.get("n_valid_frames", 0) == 0:
                continue

            folder = str(seq.get("folder"))
            exercise = str(seq.get("exercise"))
            frames = seq.get("frames", [])

            for fr in frames:
                fid = fr.get("id", None)
                ang = None
                # El ángulo puede venir en fr["joints"]["angle"] o similar; adaptamos:
                joints = fr.get("joints", {})
                ang = joints.get("angle", None)
                # Descarta nulos
                if fid is None or ang is None or ang == 0.0:
                    skipped += 1
                    continue
                labels[(folder, exercise, int(fid))] = float(ang)
                kept += 1

        print(f"[metadata] indexed {kept} valid angles, skipped {skipped}")

        # 3) Escanear disco y generar manifest por split
        root = os.path.join(cfg.data.dataset_root, cfg.data.modality)
        zfill = int(cfg.data.frame_zfill)
        ext = cfg.data.file_ext.lower().lstrip(".")
        out_dir = "data/manifests"
        out_path = os.path.join(out_dir, cfg.project.name, cfg.data.modality, cfg.data.split_name, cfg.project.run_name)
        os.makedirs(out_path, exist_ok=True)
        out_csv = os.path.join(out_path, f"{os.path.splitext(cfg.data.split_name)[0]}_{active_split}.csv")

        rows, missed = [], 0
        for subj in _safe_listdir(root):
            if subj not in allowed_subjects:
                continue

            folder_key = map_subject(subj, cfg)
            d_subj = os.path.join(root, subj)

            for ex in _safe_listdir(d_subj):
                # filtra por ejercicios si inc_ex está definido
                if inc_ex is not None and ex not in inc_ex:
                    continue
                d_ex = os.path.join(d_subj, ex)

                for cam in _safe_listdir(d_ex):
                    # filtra por cámaras si inc_cam está definido
                    if inc_cam is not None and cam not in inc_cam:
                        continue
                    d_cam = os.path.join(d_ex, cam)
                    
                    frames = sorted(
                        [p for p in glob.glob(os.path.join(d_cam, "*")) if p.lower().endswith(("."+ext,))],
                        key=_nat_key
                    )
                    for p in frames:
                        fname = os.path.basename(p)
                        stem, _ = os.path.splitext(fname)
                        m = re.findall(r"\d+", stem)
                        if not m:
                            continue
                        fid_int = int(m[-1])
                        # ojo: 'ex' ya es str de la carpeta; el índice usa str(ex)
                        ang = labels.get((folder_key, str(ex), fid_int), None)
                        if ang is None:
                            missed += 1
                            continue

                        rows.append((subj, ex, cam, fid_int, p, ang))

        normalization_function = get_normalization_function(cfg.data.normalization.function)
        split_normalizer = cfg.data.get("split_normalizer", cfg.data.split_normalizer)
        print(f"[manifest] using normalization function: {cfg.data.normalization.function}")
        
        if normalization_function is not None:
            angles = [row[5] for row in rows]
            norm_params_path = os.path.join(out_path, f"{os.path.splitext(cfg.data.split_name)[0]}_norm_params.json")

            if split_normalizer == active_split:
                print(f"[manifest] computing normalization params from current split '{active_split}'")
                norm_data, norm_params = normalization_function(np.array(angles))

                # out_norm_params = os.path.join(out_path, f"{os.path.splitext(cfg.data.split_name)[0]}_norm_params.json")
                with open(norm_params_path, "w") as f:
                    json.dump(norm_params, f)
                print(f"[manifest] applied normalization and stored {norm_params_path}")
            else:
                # norm_params_path = os.path.join(out_path, f"{os.path.splitext(cfg.data.split_name)[0]}_norm_params.json")
                print(f"[manifest] loading normalization params from split '{split_normalizer}' at {norm_params_path}")
                with open(norm_params_path, "r") as f:
                    norm_params = json.load(f)
                norm_data, _ = normalization_function(np.array(angles), params=norm_params)

            # Agregar una nueva columna al manifest con los ángulos normalizados
            for i in range(len(rows)):
                row = list(rows[i])
                row.append(float(norm_data[i]))
                rows[i] = tuple(row)

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject","exercise","camera","frame_id","img_path","angle_deg", "angle_norm"])
            w.writerows(rows)

        print(f"[manifest] wrote {len(rows)} rows to {out_csv} (missed {missed})")
        print("Done.")

def _safe_listdir(path):
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        return []

def _nat_key(p):
    b = os.path.basename(p)
    m = re.findall(r"\d+", b)
    return (int(m[-1]) if m else 10**9, b)

if __name__ == "__main__":
    main()
