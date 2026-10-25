import numpy as np
import pandas as pd
from pathlib import Path
from config import POINTS, ANGLES

def output_folder():
    root = Path(__file__).resolve().parent
    i = 1
    while True:
        folder = root / f'diagnostic_results_v{i}'
        try:
            folder.mkdir(parents=True, exist_ok=False)
            return folder
        except FileExistsError:
            i += 1

def make_summary(df, chosen, subj, exercise, camera, item, common, images):
    show = lambda keys: ','.join(f'{side}_{limb}' for side, limb in sorted(keys)) or 'unknown'
    summary = dict(subject=subj, exercise=exercise, camera=camera, local_choice=show(item['local']), consensus=show(common), selected=show(chosen),
        frames=int(df.frame.nunique()), images=images, active_count=len(chosen), selection_status=('selected' if chosen else 'unknown'))
    torso = df.drop_duplicates('frame').torso_status
    known = torso.ne('UNKNOWN')
    summary['torso_stable_pct'] = (float(100 * torso[known].eq('STABLE').mean()) if known.any() else np.nan)
    summary['torso_coverage_pct'] = float(100 * known.mean())
    summary['selected_valid_pct'] = np.nan
    summary['angle_jumps'] = np.nan
    diagnostics, percentages, jumps = [], [], []

    for side in ('left', 'right'):
        for limb in ('arm', 'leg'):
            key = side, limb
            prefix = f'{side}_{limb}'
            info = item['scores'].get(key, {})
            summary[prefix + '_active'] = key in chosen
            summary[prefix + '_score'] = info.get('score', np.nan)
            summary[prefix + '_coverage'] = info.get('coverage', np.nan)

            if key not in chosen:
                continue

            part = df[df.side == side]
            names = ANGLES[limb]
            valid = float(100 * part[limb + '_valid'].mean())
            count = int(part[[name + '_jump' for name in names]].sum().sum())
            summary[prefix + '_valid_pct'] = valid
            summary[prefix + '_angle_jumps'] = count
            percentages.append(valid)
            jumps.append(count)

            for name in names:
                summary[f'{side}_{name}_coverage_pct'] = round(float(100 * part[name + '_smooth'].notna().mean()), 2)
                summary[f'{side}_{name}_observed_pct'] = round(float(100 * part[name + '_status'].eq('observed').mean()), 2)

            for category, fields in (('point', POINTS), ('angle', names)):
                for name in fields:
                    for status, count in part[name + '_status'].value_counts().items():
                        diagnostics.append(dict(subject=subj, exercise=exercise, camera=camera, side=side, limb=limb, element=name,
                            category=category, reason=status, count=int(count)))

    if percentages:
        summary['selected_valid_pct'] = float(np.mean(percentages))
        summary['angle_jumps'] = sum(jumps)
    return summary, diagnostics


def save_tables(df, summary, diagnostics,folder, wrote, summaries, reasons):
    df.to_csv(folder / 'frame_results.csv', mode='a', header=not wrote, index=False)
    summaries.append(summary)
    reasons.extend(diagnostics)
    pd.DataFrame(summaries).to_csv(folder / 'video_summary.csv', index=False)
    cols = ['subject', 'exercise', 'camera','side', 'limb', 'element', 'category', 'reason', 'count']
    pd.DataFrame(reasons, columns=cols).to_csv(folder / 'missing_reasons.csv',index=False)
