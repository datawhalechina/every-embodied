"""Calibrate Aura's actual stereo pair from checkerboard captures.

Input: JSON list of {left, right, left_stamp, right_stamp}. Paths are relative to
the manifest. Board dimensions are INNER corners; square size is in meters.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .vision import read_image, write_json


def calibrate(manifest, columns, rows, square_m):
    if columns < 3 or rows < 3 or not math.isfinite(square_m) or not .001 <= square_m <= .2:
        raise ValueError('Require >=3x3 inner corners and square_m in [.001,.2]')
    samples = json.loads(manifest.read_text(encoding='utf-8'))
    grid = np.zeros((columns*rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    points1, points2, objects, accepted, rejected = [], [], [], [], []
    image_size = None
    for sample in samples:
        t1, t2 = sample['left_stamp'], sample['right_stamp']
        if not all(math.isfinite(t) for t in (t1, t2)) or abs(t1-t2) > .02:
            rejected.append({'sample': sample, 'reason': 'not synchronized within 20ms'})
            continue
        left, right = [read_image(manifest.parent / sample[k]) for k in ('left', 'right')]
        size = left.shape[1::-1]
        if left.shape != right.shape or image_size is not None and size != image_size:
            raise ValueError('All calibration images must have identical dimensions')
        image_size = size
        corners = [cv2.findChessboardCornersSB(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
            (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
            for im in (left, right)]
        if not all(found for found, _ in corners):
            rejected.append({'sample': sample, 'reason': 'checkerboard not found in both views'})
            continue
        points1.append(corners[0][1])
        points2.append(corners[1][1])
        objects.append(grid.copy())
        accepted.append(sample)
    if len(objects) < 12:
        raise ValueError(f'Need >=12 usable, varied synchronized board poses; found {len(objects)}')
    rms1, k1, d1, _, _ = cv2.calibrateCamera(objects, points1, image_size, None, None)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(objects, points2, image_size, None, None)
    rms, k1, d1, k2, d2, rotation, translation, _, _ = cv2.stereoCalibrate(
        objects, points1, points2, k1, d1, k2, d2, image_size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-7))
    r1, r2, p1, p2, *_ = cv2.stereoRectify(k1, d1, k2, d2, image_size, rotation, translation)
    residuals = []
    for a, b in zip(points1, points2):
        a = cv2.undistortPoints(a, k1, d1, R=r1, P=p1)
        b = cv2.undistortPoints(b, k2, d2, R=r2, P=p2)
        residuals.extend(np.abs(a[:, 0, 1]-b[:, 0, 1]).tolist())
    p95 = float(np.percentile(residuals, 95))
    result = {'image_size': list(image_size), 'K1': k1.tolist(), 'K2': k2.tolist(),
        'D1': d1.reshape(-1).tolist(), 'D2': d2.reshape(-1).tolist(), 'R': rotation.tolist(),
        'T_m': translation.reshape(-1).tolist(), 'square_m': square_m,
        'rms_px': float(rms), 'left_rms_px': float(rms1), 'right_rms_px': float(rms2),
        'epipolar_p95_px': p95, 'accepted': accepted, 'rejected': rejected,
        'quality_gate_passed': bool(max(rms, rms1, rms2) <= 1.5 and p95 <= 2.),
        'note': 'Board pose diversity, physical scale and held-out distance validation remain required'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--columns', type=int, required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--square-m', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.manifest, args.columns, args.rows, args.square_m)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('rms_px', 'epipolar_p95_px', 'quality_gate_passed')}))
    return 0 if result['quality_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
