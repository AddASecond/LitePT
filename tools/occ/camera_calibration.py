"""Image-specific camera calibration shared by Scene export and camera backfills."""
import re

import numpy as np

import static_agg as sag


def camera_names(sensors):
    """Keep both physical image variants, preferring modern raw over legacy names."""
    cameras = {}
    for name, sensor in sensors.items():
        match = re.fullmatch(r'camera(\d+)(?:_(raw|undistort))?', name)
        if not match or not isinstance(sensor, dict):
            continue
        number, variant = match.groups()
        variants = cameras.setdefault(int(number), {})
        kind = variant or 'raw'
        if kind not in variants or variant is not None:
            variants[kind] = name
    return [variants[kind] for _, variants in sorted(cameras.items())
            for kind in ('undistort', 'raw') if kind in variants]


def parse_camera(cam_doc):
    intrinsic = cam_doc['intrinsic']
    model = intrinsic.get('meta', {}).get('camera_model', 'pinhole')
    if model != 'pinhole':
        raise ValueError(f'Unsupported camera projection model: {model}')
    K = np.asarray(intrinsic['intrinsic'], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(intrinsic['distortion'], dtype=np.float64).reshape(-1)
    if distortion.size not in (0, 4, 5, 8):
        raise ValueError(f'Unsupported pinhole distortion length: {distortion.size}')
    T_v_c = np.asarray(cam_doc['extrinsic']['transformation'], dtype=np.float64).reshape(4, 4)
    if not all(np.isfinite(value).all() for value in (K, distortion, T_v_c)):
        raise ValueError('Non-finite camera calibration')
    return K, distortion, np.linalg.inv(T_v_c), T_v_c, int(intrinsic['width']), int(intrinsic['height'])


def camera_metadata(name, cam_doc, image_size, image_uri, lidar_timestamp, pose_samples):
    K, distortion, T_c_v, T_v_c, cal_w, cal_h = parse_camera(cam_doc)
    iw, ih = image_size
    if min(cal_w, cal_h, iw, ih) <= 0:
        raise ValueError('Invalid calibrated/image dimensions')
    sx, sy = iw / cal_w, ih / cal_h
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    K_image = resize @ K
    timestamp = int(cam_doc.get('timestamp') or lidar_timestamp)
    T_map_v_camera = sag.interpolate_pose_matrix(pose_samples, timestamp)
    T_map_v_lidar = sag.interpolate_pose_matrix(pose_samples, lidar_timestamp)
    variant = 'undistort' if name.endswith('_undistort') else 'raw'
    result = {
        'name': name, 'camera_id': re.sub(r'_(raw|undistort)$', '', name),
        'image_variant': variant, 'camera_model': 'pinhole',
        'distortion_model': cam_doc['intrinsic'].get('meta', {}).get('distortion_model'),
        'md5': cam_doc.get('md5'), 'timestamp': timestamp,
        'file': f'cameras/{name}.jpg', 'width': iw, 'height': ih,
        'K': K_image.reshape(-1).tolist(), 'distortion': distortion.tolist(),
        'T_c_v': T_c_v.reshape(-1).tolist(), 'T_v_c': T_v_c.reshape(-1).tolist(),
        'T_c_v_lidar_ref': (T_c_v @ np.linalg.inv(T_map_v_camera) @ T_map_v_lidar).reshape(-1).tolist(),
        'time_compensation': {'method': 'ego_pose_linear_position_nlerp_quaternion',
                              'lidar_reference_timestamp': lidar_timestamp,
                              'camera_timestamp': timestamp, 'delta_ms': (timestamp - lidar_timestamp) / 1e6},
        'image': {'uri': image_uri, 'mime': 'image/jpeg', 'width': iw, 'height': ih},
    }
    # Preserve old rectified/five-coefficient readers without pretending that
    # the first five rational coefficients describe a raw eight-coefficient lens.
    if distortion.size <= 5 or not np.any(distortion[5:]):
        result['dist5'] = np.pad(distortion[:5], (0, max(0, 5 - distortion.size))).tolist()
    return result
