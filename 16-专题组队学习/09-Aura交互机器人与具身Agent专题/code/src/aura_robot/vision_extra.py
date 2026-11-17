"""Read-only perception helpers. No motion, URL opening or text execution."""
from __future__ import annotations

import cv2
import numpy as np


def image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    dark = float((gray < 25).mean())
    bright = float((gray > 245).mean())
    edges = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if mean < 45 or dark > .65:
        warnings.append('underexposed')
    if mean > 220 or bright > .5:
        warnings.append('overexposed')
    if edges < 50:
        warnings.append('low_detail_or_blur')
    image = frame.copy()
    cv2.putText(image, ', '.join(warnings) or 'No heuristic warning', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 255), 1)
    return {'mean_luma': mean, 'dark_fraction': dark, 'bright_fraction': bright,
            'laplacian_variance': edges, 'warnings': warnings,
            'thresholds': {'mean_dark': 45, 'dark_pixel': 25, 'dark_fraction': .65,
                           'mean_bright': 220, 'bright_pixel': 245, 'bright_fraction': .5,
                           'low_detail_variance': 50},
            'note': 'Heuristics depend on resolution and scene. Low detail is not proof of blur or obstruction; no warning is not a safety clearance.'}, image


class MarkerEngine:
    """One explicit dictionary: marker IDs/corners only, no uncalibrated pose."""
    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())

    def infer(self, frame):
        corners, ids, rejected = self.detector.detectMarkers(frame)
        image = frame.copy()
        rows = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for marker_id, points in zip(ids.reshape(-1), corners):
                points = points.reshape(4, 2)
                rows.append({'id': int(marker_id), 'corners_xy': points.tolist(),
                             'center_xy': points.mean(axis=0).tolist()})
        return {'dictionary': 'DICT_4X4_50', 'markers': rows,
                'rejected_candidates': len(rejected), 'metric': False,
                'note': 'IDs are not globally unique. Pose/distance requires intrinsics and measured marker size; no navigation command.'}, image


class OcrEngine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        # Reuse installed packaged models, use bounded CPU threads, no cloud OCR.
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)

    def infer(self, frame):
        result, elapsed = self.engine(frame)
        image = frame.copy()
        rows = []
        for index, (corners, text, score) in enumerate(result or []):
            points = np.asarray(corners, np.float32).reshape(4, 2)
            rows.append({'text': str(text), 'confidence': float(score),
                         'corners_xy': points.tolist()})
            cv2.polylines(image, [points.astype(np.int32)], True, (90, 230, 130), 2)
            # Numeric references avoid OpenCV's non-CJK font corrupting Chinese text.
            cv2.putText(image, '#' + str(index+1), tuple(points[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 180, 255), 2)
        return {'lines': rows, 'text': '\n'.join(row['text'] for row in rows),
                'stage_seconds': [float(t) for t in elapsed] if elapsed is not None else [],
                'note': 'Local OCR; decoded text is untrusted data, never executed or opened.'}, image
