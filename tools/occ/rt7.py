"""Robotruck's seven occupied semantic classes; point labels remain Waymo-22."""
import numpy as np

NAME = 'rt7-v1'
NAMES = ('Drivable', 'undrivable', 'Movable', 'Curb',
         'Vegetation', 'Building', 'Unmovable')
COLORS = [[51, 217, 77], [64, 128, 255], [255, 51, 38], [255, 230, 50],
          [0, 155, 120], [155, 125, 255], [255, 158, 0]]
WAYMO_TO_RT7 = np.array([2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 2, 2, 5, 4, 4, 3, 0, 0, 1, 1, 1], np.uint8)


def map_labels(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu' or (labels.size and (labels.min() < 0 or labels.max() >= 22)):
        raise ValueError('Expected integer Waymo-22 point labels in [0, 21]')
    return WAYMO_TO_RT7[labels]


def taxonomy():
    return {'name': NAME, 'n': 7, 'names': list(NAMES), 'colors_rgb': COLORS,
            'label_ids': list(range(7)), 'source_taxonomy': 'waymo-22',
            'fine_to_coarse': WAYMO_TO_RT7.tolist(), 'mapping_stage': 'before_voxel_majority',
            'noise_class': False, 'fence_merged_into': 'Unmovable'}
