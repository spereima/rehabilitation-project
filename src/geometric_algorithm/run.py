
import json
import os
from queue import Empty
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from results import make_summary
from pipeline import process_video
from movement import motion_scores, local_choice, vote_rescue, annotate, with_weight
from config import (BASE, OUTPUT as SOURCE_OUTPUT, CAMERAS, WORKERS, SAVE_IMAGES, DEFAULT_WEIGHT, COLOR_TOLERANCE, MIN_AREA,
    MAX_JUMP_REL_SCALE, MAX_GAP_FRAMES, ANGLE_JUMP_DEG, SKIP_S, WINDOW_S, BASELINE_S, COLORS, SUBJECTS, EXERCISES)

OUTPUT = SOURCE_OUTPUT
VERSION = 'v9_vote_rescue_1'
GEOMETRY_VERSION = 'geometry_v8_threads_compare_1'
WEIGHT = DEFAULT_WEIGHT
TRUST_LEGACY_CACHE = True
ANGLES = {'leg': ('hip_angle_2d', 'knee_angle', 'ankle_angle'), 'arm': ('shoulder_angle_2d', 'elbow_angle', 'wrist_angle')}
ROM_COLUMNS = [f'{side}_{angle}_rom' for side in ('left', 'right') for limb in ANGLES for angle in ANGLES[limb]]
images = 0

def calculate_rom(df, chosen):
    result = {column: np.nan for column in ROM_COLUMNS}
    for side, limb in sorted(chosen):
        if limb not in ANGLES:
            continue
        data = df[df.side == side]
        for angle in ANGLES[limb]:
            column, output = angle + '_smooth', f'{side}_{angle}_rom'
            if column not in data.columns:
                continue
            values = pd.to_numeric(data[column], errors='coerce').to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) >= 10:
                result[output] = round(float(np.percentile(values, 95) - np.percentile(values, 5)), 2)
    return result


def cache_settings(video):
    stat = video.stat()
    return {'geometry_version': GEOMETRY_VERSION, 'color_tolerance': COLOR_TOLERANCE, 'min_area': MIN_AREA, 'max_jump_rel_scale': MAX_JUMP_REL_SCALE,
        'max_gap_frames': MAX_GAP_FRAMES, 'angle_jump_deg': ANGLE_JUMP_DEG, 'skip_s': SKIP_S, 'window_s': WINDOW_S,
        'baseline_s': list(BASELINE_S), 'colors': COLORS, 'video_size': stat.st_size, 'video_mtime_ns': stat.st_mtime_ns}


def valid_cache(folder, video):
    cache, meta = folder / 'geometry.pkl', folder / 'geometry_meta.json'
    if not cache.exists() or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding='utf-8')) == cache_settings(video)
    except (OSError, ValueError):
        return False


def read_geometry(folder, source, video):
    cache, meta = folder / 'geometry.pkl', folder / 'geometry_meta.json'
    old_cache, old_meta = source / 'geometry.pkl', source / 'geometry_meta.json'

    if valid_cache(folder, video):
        try:
            return pd.read_pickle(cache)
        except (OSError, EOFError, ValueError):
            pass

    if old_cache.exists() and (valid_cache(source, video) or TRUST_LEGACY_CACHE and not old_meta.exists()):
        try:
            return pd.read_pickle(old_cache)
        except (OSError, EOFError, ValueError):
            pass

    df = process_video(video)
    if df.empty:
        return df

    temp_cache, temp_meta = folder / 'geometry_temp.pkl', folder / 'geometry_meta_temp.json'
    df.to_pickle(temp_cache)
    temp_meta.write_text(json.dumps(cache_settings(video), indent=2), encoding='utf-8')
    os.replace(temp_cache, cache)
    os.replace(temp_meta, meta)
    return df


def completed(folder):
    marker = folder / 'done.json'
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text(encoding='utf-8'))
        return (info.get('version') == VERSION and info.get('weight') == WEIGHT and info.get('save_images') == SAVE_IMAGES
            and all((folder / f).exists() for f in ('summary.csv', 'diagnostics.csv', 'frames.csv')))
    except (OSError, ValueError):
        return False


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def job(task):
    subject, exercise, progress_queue = task
    videos = [(cam, BASE / str(subject) / exercise / f'{cam}.mp4') for cam in CAMERAS if (BASE / str(subject) / exercise / f'{cam}.mp4').exists()]
    if not videos:
        return 'missing'

    if all(completed(OUTPUT / 'per_video' / f'{subject}_{exercise}_{cam}') for cam, _ in videos):
        for _ in videos:
            progress_queue.put(1)
        return 'skipped'

    items = []
    for cam, video in videos:
        name = f'{subject}_{exercise}_{cam}'
        folder, source = OUTPUT / 'per_video' / name, SOURCE_OUTPUT / 'per_video' / name
        folder.mkdir(parents=True, exist_ok=True)
        df = read_geometry(folder, source, video)
        if df.empty:
            progress_queue.put(1)
            continue
        items.append({'camera': cam, 'file': video, 'df': df, 'scores': motion_scores(df)})

    if not items:
        return 'missing'

    weighted = []
    for item in items:
        scores = with_weight(item['scores'], WEIGHT)
        weighted.append({**item, 'scores': scores, 'local': local_choice(scores)})
    common = vote_rescue(weighted) if len(weighted) >= 3 else set()

    for item in weighted:
        cam, video = item['camera'], item['file']
        folder = OUTPUT / 'per_video' / f'{subject}_{exercise}_{cam}'
        if completed(folder):
            progress_queue.put(1)
            continue

        df, chosen = annotate(item['df'], item, common)
        df['subject'], df['exercise'], df['camera'] = subject, exercise, cam
        summary, diagnostics = make_summary(df, chosen, subject, exercise, cam, item, common, images)

        for key in ('repetitions_total', 'exercise_repetitions', 'repetitions_by_side', 'confirmed_by_both', 'confirmed_by_one',
            'expected_per_side', 'all_active_sides_have_4'):
            summary.pop(key, None)

        summary['active_limbs_count'] = len(chosen)
        summary.update(calculate_rom(df, chosen))
        df.to_csv(folder / 'frames.csv', index=False)
        save_csv([summary], folder / 'summary.csv')
        save_csv(diagnostics, folder / 'diagnostics.csv')
        (folder / 'done.json').write_text(json.dumps({
            'ok': True, 'version': VERSION, 'weight': WEIGHT, 'save_images': SAVE_IMAGES, 'repetitions_enabled': False
        }), encoding='utf-8')
        progress_queue.put(1)

    return 'processed'


def aggregate():
    base = OUTPUT / 'per_video'
    for source, target in (('summary.csv', 'video_summary.csv'), ('diagnostics.csv', 'missing_reasons.csv')):
        tables = []
        for subject in SUBJECTS:
            for exercise in EXERCISES:
                for cam in CAMERAS:
                    folder = base / f'{subject}_{exercise}_{cam}'
                    if not completed(folder):
                        continue
                    file = folder / source
                    if not file.exists():
                        continue
                    try:
                        data = pd.read_csv(file)
                        if not data.empty:
                            tables.append(data)
                    except (pd.errors.EmptyDataError, OSError, ValueError):
                        continue

        output = OUTPUT / target
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(output, index=False)
        elif output.exists():
            output.unlink()


def main():
    if not BASE.is_dir():
        raise FileNotFoundError(f'Check BASE in config.py: {BASE}')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tasks = [(subject, exercise) for subject in SUBJECTS for exercise in EXERCISES]
    total = sum((BASE / str(subject) / exercise / f'{cam}.mp4').exists() for subject, exercise in tasks for cam in CAMERAS)
    counts = {'processed': 0, 'skipped': 0, 'missing': 0, 'errors': 0}

    print('V8 source:', SOURCE_OUTPUT)
    print('V9 output:', OUTPUT)
    print('Videos:', total)

    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(job, (*task, progress_queue)): task for task in tasks}
            pending = set(futures)

            with tqdm(total=total, desc='Processing', unit='video', dynamic_ncols=True) as progress:
                while pending:
                    try:
                        progress.update(progress_queue.get(timeout=.5))
                    except Empty:
                        pass

                    finished = {future for future in pending if future.done()}
                    for future in finished:
                        try:
                            counts[future.result()] += 1
                        except Exception as exc:
                            counts['errors'] += 1
                            tqdm.write(f'ERROR {futures[future]}: {exc!r}')
                    pending -= finished

                while True:
                    try:
                        progress.update(progress_queue.get_nowait())
                    except Empty:
                        break

    aggregate()
    print('Finished:', OUTPUT)
    print('Summary:', counts)


if __name__ == '__main__':
    main()
