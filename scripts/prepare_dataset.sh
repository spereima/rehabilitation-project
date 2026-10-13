#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ----------------------------------------------------------
# Usage check
# ----------------------------------------------------------
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <input_root_mp4> <modality> <output_root_frames>"
  echo "  <input_root_mp4>    Dataset root with Mp4 files"
  echo "  <modality>          sils | of_tvl1 | of_gmflow | semantic_segmentation"
  echo "  <output_root_frames> Root where PNGs will be saved"
  exit 1
fi

INPUT_ROOT="$1"
MODALITY="$2"
OUTPUT_ROOT="$3"

# ----------------------------------------------------------
# Validar modalidad
# ----------------------------------------------------------
case "$MODALITY" in
  sils|sils_cropped|of_tvl1|of_gmflow|semantic_segmentation)
    ;;
  *)
    echo "[ERROR] Modality no válida: $MODALITY"
    echo "       Usa: sils, of_tvl1, of_gmflow o semantic_segmentation"
    exit 1
    ;;
esac

IN_MOD_DIR="${INPUT_ROOT%/}/$MODALITY"

if [[ ! -d "$IN_MOD_DIR" ]]; then
  echo "[ERROR] Folder not found for modality: $IN_MOD_DIR"
  exit 1
fi

echo "[INFO] Input root (MP4):   $INPUT_ROOT"
echo "[INFO] Modality:           $MODALITY"
echo "[INFO] Output root (PNG):  $OUTPUT_ROOT"
echo

# ----------------------------------------------------------
# Subjects, exercises and cameras loop
# ----------------------------------------------------------
for subj_dir in "$IN_MOD_DIR"/*; do
  [[ -d "$subj_dir" ]] || continue
  subj_id="$(basename "$subj_dir")"

  for ex_dir in "$subj_dir"/*; do
    [[ -d "$ex_dir" ]] || continue
    # ex_id="$(basename "$ex_dir")"

    # # Find MP4 files
    # videos=( "$ex_dir"/*.mp4 )
    ex_id="$(basename "$ex_dir")"

    # Process only exercises 01 and 05
    if [[ "$ex_id" != "01" && "$ex_id" != "05" ]]; then
      continue
    fi

    # Find MP4 files
    videos=( "$ex_dir"/*.mp4 )
    if (( ${#videos[@]} == 0 )); then
      echo "[WARN] MP4 not found in: $ex_dir"
      continue
    fi

    for video_path in "${videos[@]}"; do
      # cam_file="$(basename "$video_path")"
      # cam_id="${cam_file%.mp4}"
      cam_file="$(basename "$video_path")"
      cam_id="${cam_file%.mp4}"

      # Process only cam0
      if [[ "$cam_id" != "cam0" ]]; then
        continue
      fi

      # Output structure:
      # <OUTPUT_ROOT>/<MODALITY>/<subject_id>/<exercise_id>/<cam_id>/*.png
      out_dir="${OUTPUT_ROOT%/}/$MODALITY/$subj_id/$ex_id/$cam_id"
      mkdir -p "$out_dir"

      # If already PNGs, skip
      if compgen -G "$out_dir/*.png" > /dev/null; then
        echo "[INFO] Already found PNG, skipping: $out_dir"
        continue
      fi

      echo "[INFO] Extracting frames: $video_path -> $out_dir"

      ffmpeg -y -i "$video_path" -start_number 0 "$out_dir/%04d.png"
    done
  done
done

echo
echo "[INFO] Process complete."
