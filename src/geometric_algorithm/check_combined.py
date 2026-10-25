
import pandas as pd
from tqdm import tqdm
from config import OUTPUT, EXPECTED
from combined_repetitions import SYNC_EXERCISES
from combined_repetitions import combine_repetitions

SOURCE = OUTPUT / 'repetitions_candidate.csv'
SUMMARY = OUTPUT / 'video_summary.csv'

reps = pd.read_csv(SOURCE)
videos = pd.read_csv(SUMMARY)
results = []
checks = []

for _, video in tqdm(videos.iterrows(), total=len(videos), desc='Combining'):
    subject = int(video['subject'])
    exercise = str(video['exercise']).zfill(2)
    camera = str(video['camera'])

    selected = str(video.get('selected', ''))
    chosen = {tuple(item.split('_', 1)) for item in selected.split(',') if '_' in item}
    current = reps[(reps['subject'] == subject) & (reps['exercise'].astype(str).str.zfill(2) == exercise) &(reps['camera'] == camera)]
    rows = combine_repetitions(current.to_dict('records'), chosen, subject, exercise, camera)

    results.extend(rows)
    expected_limb, expected_sides = EXPECTED[exercise]
    expected = {(side, expected_limb) for side in expected_sides}

    checkable = (exercise in SYNC_EXERCISES or len(expected_sides) == 1)

    checks.append({
        'subject': subject,
        'exercise': exercise,
        'camera': camera,
        'selected_correctly': chosen == expected,
        'combined_repetitions': len(rows),
        'both_limbs_confirmed': sum(r['confidence'] == 'both_limbs' for r in rows),
        'one_limb_only': sum(r['confidence'] == 'one_limb_only' for r in rows),
        'expected_repetitions': 4 if checkable else None,
        'absolute_error': (abs(len(rows) - 4) if checkable and chosen == expected else None)})

pd.DataFrame(results).to_csv(OUTPUT / 'combined_repetitions.csv', index=False)

check = pd.DataFrame(checks)
check.to_csv(OUTPUT / 'combined_check.csv', index=False)
valid = check[check['absolute_error'].notna()]
