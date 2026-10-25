import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.getenv('REHAB_DATA_DIR', ROOT / 'data_check' / 'semantic_segmentation' / 'semantic_segmentation'))
OUTPUT = Path(os.getenv('REHAB_OUTPUT_DIR', ROOT / 'src' / 'geometric_algorithm' / 'results_all'))
SUBJECTS = list(range(27))
EXERCISES = [f'{i:02d}' for i in range(1, 17)]
CAMERAS = [f'cam{i}' for i in range(5)]
WORKERS = 2
SAVE_IMAGES = False
SAVE_EVERY = 90

SKIP_S = 3.5
REPETITION_START_S = 0.0
WINDOW_S = 5.0
BASELINE_S = (3.5, 4.5)

TORSO_LIMIT = 10.0
COLOR_TOLERANCE = 9
MIN_AREA = 30
MIN_COVERAGE = 0.20
MIN_MOTION = 0.06
MAX_GAP_FRAMES = 2
MAX_JUMP_REL_SCALE = 0.20
ANGLE_JUMP_DEG = 30
REQUIRE_CONSENSUS_EVIDENCE = False

DEFAULT_WEIGHT = 0.7
WEIGHTS = [round(i / 10, 1) for i in range(5, 11)]

COLORS = {'torso': '#ff0000', 'head': '#ffffff', 'right_hand': '#0000ff', 'left_hand': '#00ff00', 'right_foot': '#800000',
    'left_foot': '#ffff00', 'right_thigh': '#808000', 'left_thigh': '#008080','right_shin': '#800080', 'left_shin': '#808080',
    'right_upper_arm': '#ff0080', 'left_upper_arm': '#ff8000', 'right_lower_arm': '#80ff80', 'left_lower_arm': '#ff80ff'}

CHAINS = {'leg': ('hip', 'knee', 'ankle', 'toe'), 'arm': ('shoulder', 'elbow', 'wrist', 'hand')}
TRIPLES = {'hip_angle_2d': ('torso_top', 'hip', 'knee'), 'knee_angle': ('hip', 'knee', 'ankle'), 'ankle_angle': ('knee', 'ankle', 'toe'),
    'shoulder_angle_2d': ('torso_bottom', 'shoulder', 'elbow'), 'elbow_angle': ('shoulder', 'elbow', 'wrist'), 'wrist_angle': ('elbow', 'wrist', 'hand')}
ANGLES = {'leg': tuple(list(TRIPLES)[:3]), 'arm': tuple(list(TRIPLES)[3:])}

POINTS = tuple(dict.fromkeys(joint for chain in CHAINS.values() for joint in chain)) + ('torso_top', 'torso_bottom')

SEGMENT_OF = {'hip': 'thigh', 'knee': 'thigh+shin', 'ankle': 'shin+foot', 'toe': 'foot', 'shoulder': 'upper_arm', 'elbow': 'upper_arm+lower_arm',
    'wrist': 'lower_arm+hand', 'hand': 'hand', 'torso_top': 'torso', 'torso_bottom': 'torso'}

EXPECTED = {'01': ('leg', {'left'}), '02': ('leg', {'left', 'right'}), '03': ('leg', {'left'}), '04': ('leg', {'left'}), '05': ('leg', {'right'}),
    '06': ('leg', {'left', 'right'}), '07': ('leg', {'right'}), '08': ('leg', {'right'}), '09': ('arm', {'left', 'right'}), '10': ('arm', {'left', 'right'}),
    '11': ('arm', {'left', 'right'}), '12': ('arm', {'left'}), '13': ('arm', {'left', 'right'}), '14': ('arm', {'left', 'right'}), '15': ('arm', {'left', 'right'}),
    '16': ('arm', {'right'})
}
