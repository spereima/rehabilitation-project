from pathlib import Path
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as TF
from scipy.signal import savgol_filter
from src.models import get_model

ROOT = Path(__file__).resolve().parent
if not (ROOT / "conf").exists():
    ROOT = ROOT.parent

st.set_page_config(page_title="Аналіз руху", layout="wide")
st.markdown("""<style>
.block-container {max-width:1050px;padding-top:2rem}
[data-testid="stMetric"] {border-bottom:1px solid #d6d6d6;padding:10px 0}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Завантаження моделі...")
def load_model(modality):
    model_key = "baseline_sils_resnet18" if modality == "sils" else "baseline_rgb_resnet18"
    config = OmegaConf.load(ROOT / "conf" / "model" / f"{model_key}.yaml")
    model = get_model(config)(config)
    ckpt = ROOT / "models" / modality / "epoch_3.pth"
    norm = ROOT / "models" / modality / "final_train_norm_params.json"
    if not ckpt.exists() or not norm.exists():
        raise FileNotFoundError(f"Не знайдено модель або нормалізацію: {ckpt}, {norm}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    with norm.open() as f:
        stats = json.load(f)
    return model, device, float(stats["mean"]), float(stats["std"])


def calculate_descriptors(frame_ids, angles, fps):
    frames = np.asarray(frame_ids, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 3:
        return None
    dt = np.diff(frames) / fps
    valid = dt > 0
    velocity = np.diff(angles)[valid] / dt[valid]
    if len(velocity) < 2:
        return None
    vel_times = (frames[:-1][valid] + frames[1:][valid]) / (2 * fps)
    acceleration = np.diff(velocity) / np.diff(vel_times)
    return {
        "ROM (°)": float(np.max(angles) - np.min(angles)),
        "Середня швидкість (°/с)": float(np.mean(np.abs(velocity))),
        "Пікова швидкість (°/с)": float(np.max(np.abs(velocity))),
        "Плавність (proxy, °/с²)": float(np.mean(np.abs(acceleration))) if len(acceleration) else float("nan"),
    }



def smooth_signal(angles, window=11):
    """Smooth only contiguous finite stretches, leaving original predictions intact."""
    values = np.asarray(angles, dtype=float)
    output = values.copy()
    finite = np.isfinite(values)
    cuts = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    for begin, end in zip(cuts[::2], cuts[1::2]):
        length = end - begin
        width = min(int(window), length)
        if width % 2 == 0:
            width -= 1
        if width >= 5:
            output[begin:end] = savgol_filter(values[begin:end], width, 2)
    return output


def detect_repetitions(frame_ids, angles, fps, min_duration_s=0.7,
                       prominence_deg=8.0, cycle_type="auto",
                       baseline_frames=12, return_tolerance_deg=8.0):
    """Detect full excursions from a resting position and back."""
    from scipy.ndimage import gaussian_filter1d, median_filter

    frames = np.asarray(frame_ids, dtype=int)
    values = np.asarray(angles, dtype=float)

    good = np.isfinite(values)
    frames, values = frames[good], values[good]

    if len(frames) < 9:
        return [], [], None

    splits = np.split(
        np.arange(len(frames)),
        np.flatnonzero(np.diff(frames) != 1) + 1
    )

    complete = []
    incomplete = []
    directions = []

    min_frames = max(1, round(min_duration_s * fps))

    for indices in splits:
        if len(indices) < 9:
            continue

        fr = frames[indices]
        v = values[indices]

        median_window = max(3, round(fps * 0.2))
        if median_window % 2 == 0:
            median_window += 1

        smooth = median_filter(v, size=median_window)
        smooth = gaussian_filter1d(
            smooth,
            sigma=max(1.0, fps * 0.25)
        )

        baseline_count = max(
            1, min(int(baseline_frames), len(smooth))
        )
        baseline = float(np.median(smooth[:baseline_count]))

        difference = smooth - baseline

        if cycle_type == "valley":
            direction = 1
        elif cycle_type == "peak":
            direction = -1
        else:
            upward = np.percentile(difference, 95)
            downward = -np.percentile(difference, 5)

            direction = 1 if upward >= downward else -1

        excursion = direction * difference

        amplitude = float(
            np.percentile(excursion, 95)
        )

        if amplitude < prominence_deg:
            continue

        directions.append(direction)

        enter_threshold = max(
            float(prominence_deg),
            amplitude * 0.4
        )

        peak_threshold = max(
            float(prominence_deg),
            amplitude * 0.65
        )

        return_threshold = min(
            float(return_tolerance_deg),
            amplitude * 0.15
        )

        active = False
        start = 0
        maximum = 0.0

        for i in range(len(fr)):
            current = excursion[i]

            if not active:
                if current >= enter_threshold:
                    active = True
                    maximum = current

                    start = i
                    while (
                        start > 0
                        and excursion[start - 1] > return_threshold
                    ):
                        start -= 1

            else:
                maximum = max(maximum, current)

                if current <= return_threshold:
                    duration = fr[i] - fr[start]

                    if (
                        duration >= min_frames
                        and maximum >= peak_threshold
                    ):
                        complete.append(
                            (int(fr[start]), int(fr[i]))
                        )

                    active = False
                    maximum = 0.0

        if active and maximum >= peak_threshold:
            incomplete.append(
                (int(fr[start]), int(fr[-1]))
            )

    if cycle_type == "auto":
        if directions:
            chosen = "valley" if directions[0] == 1 else "peak"
        else:
            chosen = None
    else:
        chosen = cycle_type

    return complete, incomplete, chosen

def repetition_plot(results, boundaries, reference, incomplete=None):
    plot_cols = ["frame_id", "time_s", "angle_pred"]
    if "angle_gt" in results:
        plot_cols.append("angle_gt")
    source = results[plot_cols].melt(id_vars=["frame_id", "time_s"],
                                    value_vars=[c for c in plot_cols if c.startswith("angle_")],
                                    var_name="Серія", value_name="Кут (°)").dropna()
    line = alt.Chart(source).mark_line(strokeWidth=2).encode(
        x=alt.X("time_s:Q", title="Час (с)"),
        y=alt.Y("Кут (°):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Серія:N", scale=alt.Scale(
            domain=["angle_gt", "angle_pred"], range=["#907fdd", "#f05050"])))
    if not boundaries and not incomplete:
        return line.properties(height=320).interactive()
    marks = []
    for start, end in boundaries:
        marks.extend([{"time_s": start / float(results["fps_internal"].iloc[0]), "Тип": "Повне"},
                      {"time_s": end / float(results["fps_internal"].iloc[0]), "Тип": "Повне"}])
    for start, end in (incomplete or []):
        marks.extend([{"time_s": start / float(results["fps_internal"].iloc[0]), "Тип": "Неповне"},
                      {"time_s": end / float(results["fps_internal"].iloc[0]), "Тип": "Неповне"}])
    marks = pd.DataFrame(marks).drop_duplicates()
    rules = alt.Chart(marks).mark_rule(strokeDash=[5,4], opacity=0.7).encode(
        x="time_s:Q", color=alt.Color("Тип:N", scale=alt.Scale(
            domain=["Повне", "Неповне"], range=["#907fdd", "#f05050"])))
    return (line + rules).properties(height=320).interactive()


def repetition_table(results, fps, boundaries):
    rows = []
    for number, (start, end) in enumerate(boundaries, start=1):
        section = results[(results["frame_id"] >= start) &
                          (results["frame_id"] <= end)]
        pred = calculate_descriptors(section["frame_id"], section["angle_pred_processed"], fps)
        if pred is None:
            continue
        row = {"Повторення": number, "Початковий кадр": start,
               "Кінцевий кадр": end, "Тривалість (с)": (end - start) / fps,
               "ROM прогноз (°)": pred["ROM (°)"],
               "Середня швидкість прогноз (°/с)": pred["Середня швидкість (°/с)"],
               "Плавність прогноз (°/с²)": pred["Плавність (proxy, °/с²)"]}
        if "angle_gt" in section and section["angle_gt"].notna().all():
            gt = calculate_descriptors(section["frame_id"], section["angle_gt"], fps)
            if gt is not None:
                row["ROM GT (°)"] = gt["ROM (°)"]
                row["Помилка ROM (°)"] = abs(pred["ROM (°)"] - gt["ROM (°)"])
                row["MAE кута (°)"] = float((section["angle_pred"] - section["angle_gt"]).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def predict_video(video_path, modality, model, device, mean, std, progress):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Не вдалося відкрити MP4")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Неможливо визначити FPS відео")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids, predictions, batch = [], [], []
    batch_ids = []
    frame_id = 0

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            values = model(x).float().reshape(-1).cpu().numpy() * std + mean
        predictions.extend(values.tolist())
        frame_ids.extend(batch_ids)
        batch.clear()
        batch_ids.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if modality == "sils":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = Image.fromarray(frame).convert("L")
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame).convert("RGB")
            image = image.resize((224, 224), Image.Resampling.NEAREST)
            batch.append(TF.to_tensor(image))
            batch_ids.append(frame_id)
            if len(batch) == 16:
                flush()
            frame_id += 1
            if total > 0 and frame_id % 30 == 0:
                progress.progress(min(frame_id / total, 1.0))
        flush()
    finally:
        cap.release()
    if not frame_ids:
        raise ValueError("Відео не містить кадрів")
    progress.progress(1.0)
    return pd.DataFrame({"frame_id": frame_ids, "angle_pred": predictions}), fps


def read_ground_truth(upload, subject, exercise):
    if upload is not None:
        data = pd.read_csv(upload)
        angle_col = next((c for c in ("angle_gt", "angle_deg") if c in data.columns), None)
        if "frame_id" not in data.columns or angle_col is None:
            raise ValueError("CSV повинен мати frame_id і angle_gt або angle_deg")
        return data[["frame_id", angle_col]].rename(columns={angle_col: "angle_gt"})
    if subject is None or exercise is None:
        return None
    path = ROOT / "dataset_check" / "dataset_3d_with_angles.json"
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено {path}")
    with path.open() as f:
        content = json.load(f)
    rows = []
    for seq in content.get("data", content):
        if str(seq.get("folder")) != str(subject):
            continue
        if int(seq.get("exercise", -1)) != int(exercise):
            continue
        for frame in seq.get("frames") or []:
            angle = (frame.get("joints") or {}).get("angle")
            if frame.get("id") is not None and angle is not None and angle != 0:
                rows.append({"frame_id": int(frame["id"]), "angle_gt": float(angle)})
    if not rows:
        raise ValueError("Немає валідних GT-кадрів для вибраної вправи та учасника")
    return pd.DataFrame(rows)


st.title("Аналіз руху")
st.caption("Оцінка руху за відео")
st.subheader("Вхідне відео")
upload = st.file_uploader("Завантаж MP4 із силуетом або семантичною сегментацією", type=["mp4"])
modality_label = st.selectbox("Тип відео", ["Силует", "Семантична сегментація"])
modality = "sils" if modality_label == "Силует" else "semantic_segmentation"
st.markdown("**Еталонні дані (необов’язково)**")
gt_source = st.radio("Джерело еталонних кутів", ["Немає", "CSV", "Датасет (учасник + вправа)"], horizontal=True)
gt_csv, subject, exercise = None, None, None
if gt_source == "CSV":
    gt_csv = st.file_uploader("CSV з колонками frame_id, angle_gt", type=["csv"])
elif gt_source.startswith("Датасет"):
    col1, col2 = st.columns(2)
    subject = col1.number_input("Учасник", min_value=0, step=1, value=0)
    exercise = col2.number_input("Вправа", min_value=1, max_value=16, step=1, value=1)
    st.info("Вибране відео має бути саме з цієї вправи й учасника, а нумерація кадрів — збігатися з JSON.")

if st.button("Аналізувати відео", type="primary", disabled=upload is None):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            temp.write(upload.getbuffer())
            temp_path = Path(temp.name)
        model, device, mean, std = load_model(modality)
        progress = st.progress(0.0)
        with st.spinner("Обробляємо кадри..."):
            predictions, fps = predict_video(temp_path, modality, model, device, mean, std, progress)
            gt = read_ground_truth(gt_csv, subject, exercise)
        if gt is not None:
            gt = gt.copy()
            gt["frame_id"] = pd.to_numeric(gt["frame_id"], errors="coerce")
            gt["angle_gt"] = pd.to_numeric(gt["angle_gt"], errors="coerce")
            gt = gt.dropna().drop_duplicates("frame_id")
            results = predictions.merge(gt, on="frame_id", how="left", validate="one_to_one")
        else:
            results = predictions
        results["time_s"] = results["frame_id"] / fps
        st.session_state["analysis"] = (results, fps, modality)
    except Exception as exc:
        st.error(f"Помилка: {exc}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
if "analysis" in st.session_state:
    results, fps, saved_modality = st.session_state["analysis"]
    st.markdown("**Налаштування обробки руху**")
    apply_smoothing = st.checkbox(
        "Згладжувати прогноз для ROM, швидкості та плавності",
        value=True,
        help="Не змінює оригінальні прогнози, MAE, RMSE чи графік помилок.")
    smoothing_window = st.slider("Вікно згладжування (кадри)", 5, 31, 11, 2,
                                 disabled=not apply_smoothing)
    results = results.copy()
    results["angle_pred_processed"] = (smooth_signal(results["angle_pred"], smoothing_window)
                                        if apply_smoothing else results["angle_pred"])
    results["fps_internal"] = fps
    st.subheader("Результати")
    st.caption(f"Оброблено кадрів: {len(results)}")
    gt_available = "angle_gt" in results and results["angle_gt"].notna().any()
    pred_stats = calculate_descriptors(results["frame_id"], results["angle_pred_processed"], fps)
    if pred_stats:
        c1, c2 = st.columns(2)
        c1.metric("Діапазон руху", f"{pred_stats['ROM (°)']:.1f}°")
        c2.metric("Середня швидкість руху", f"{pred_stats['Середня швидкість (°/с)']:.1f}°/с")
    if gt_available:
        matched = results.dropna(subset=["angle_gt"])
        error = matched["angle_pred"] - matched["angle_gt"]
        st.metric("Середня похибка кута", f"{error.abs().mean():.2f}°",
                  help="Середня різниця між прогнозом моделі та еталонними значеннями")
        st.caption(f"Еталонні значення доступні для {len(matched)} із {len(results)} кадрів")
        gt_stats = calculate_descriptors(matched["frame_id"], matched["angle_gt"], fps)
        pred_matched_stats = calculate_descriptors(matched["frame_id"], matched["angle_pred_processed"], fps)
        if gt_stats and pred_matched_stats:
            compare = pd.DataFrame({"Ground Truth": gt_stats, "Predicted": pred_matched_stats})
            compare["Абсолютна помилка"] = (compare["Ground Truth"] - compare["Predicted"]).abs()
            st.dataframe(compare, use_container_width=True)
        if len(matched) != len(results):
            st.warning("Частина кадрів не має Ground Truth. Метрики розраховано лише для кадрів із GT.")
    else:
        st.caption("Ground Truth відсутній: відображаються лише прогнози та їхні характеристики.")
    
    if gt_available:
        st.subheader("Порівняння прогнозу з еталонними значеннями")
        st.caption(
            "Фіолетова лінія — еталонний кут із датасету, "
            "червона — кут, визначений моделлю."
        )

        comparison = results[["time_s", "angle_gt", "angle_pred"]].copy()

        comparison = comparison.rename(
            columns={
                "angle_gt": "Еталонний кут",
                "angle_pred": "Прогноз моделі"
            }
        )

        comparison = comparison.melt(
            id_vars="time_s",
            var_name="Джерело",
            value_name="Кут (°)"
        ).dropna()

        chart = alt.Chart(comparison).mark_line(
            strokeWidth=2
        ).encode(
            x=alt.X("time_s:Q", title="Час (с)"),
            y=alt.Y(
                "Кут (°):Q",
                scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "Джерело:N",
                scale=alt.Scale(
                    domain=["Еталонний кут", "Прогноз моделі"],
                    range=["#907fdd", "#f05050"]
                ),
                legend=alt.Legend(title=None)
            ),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Час (с)", format=".2f"),
                alt.Tooltip("Джерело:N"),
                alt.Tooltip("Кут (°):Q", format=".2f")
            ]
        ).properties(
            height=320
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
    st.subheader("Визначення повторень")
    gt_complete = "angle_gt" in results and results["angle_gt"].notna().all()
    if gt_available and not gt_complete:
        st.warning("GT є не для всіх кадрів. Для визначення повторень використаємо прогноз, щоб не поєднувати несуміжні фрагменти GT.")
    reference = "angle_gt" if gt_complete else "angle_pred_processed"
    st.caption("Повторення визначаються автоматично за зміною кута. Результат можна перевірити на графіку.")
    with st.expander("Налаштування визначення повторень"):
        cycle_label = st.selectbox("Початкове положення", ["Автоматично", "Низький кут → високий → низький", "Високий кут → низький → високий"])
        cycle_type = {"Автоматично":"auto", "Низький кут → високий → низький":"valley",
                      "Високий кут → низький → високий":"peak"}[cycle_label]
        prominence = st.slider("Мінімальна амплітуда повторення (°)", 3, 40, 8, 1)
        min_duration = st.slider("Мінімальна тривалість повторення (с)", 0.3, 4.0, 0.7, 0.1)
        baseline_frames = st.slider("Кадрів для оцінки початкового положення", 3, 30, 12, 1)
        return_tolerance = st.slider("Допуск повернення до початкового кута (°)", 2, 15, 8, 1)
    boundaries, incomplete, chosen = detect_repetitions(
        results["frame_id"], results[reference], fps,
        min_duration_s=min_duration, prominence_deg=prominence, cycle_type=cycle_type,
        baseline_frames=baseline_frames, return_tolerance_deg=return_tolerance)
    if chosen:
        st.caption(f"Визначений напрямок: {'від меншого кута до більшого' if chosen == 'valley' else 'від більшого кута до меншого'}")
    st.altair_chart(repetition_plot(results, boundaries, reference, incomplete), use_container_width=True)
    count_col, partial_col = st.columns(2)
    count_col.metric("Повних повторень", len(boundaries))
    partial_col.metric("Незавершених рухів", len(incomplete))
    st.caption("Пунктирні лінії показують межі рухів. Підрахунок приблизний.")
    reps = repetition_table(results, fps, boundaries)
    if len(reps):
        st.dataframe(reps, use_container_width=True, hide_index=True)
        st.download_button("Завантажити повторення CSV",
                           reps.to_csv(index=False).encode("utf-8"),
                           "rehab_repetitions.csv", "text/csv")
    else:
        st.info("Повторення не виявлено. Спробуй інший тип циклу або меншу амплітуду.")

    st.download_button("Завантажити прогнози CSV", results.to_csv(index=False).encode("utf-8"), "rehab_predictions.csv", "text/csv")
