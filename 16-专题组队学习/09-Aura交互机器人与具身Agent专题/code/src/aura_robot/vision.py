"""Local Aura vision workbench. Perception only; no actuator or cloud calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .debug import AuraClient
from .lightnav import fetch_stream_frame

TASKS = ('detect', 'pose', 'segment', 'faces', 'qr', 'depth', 'ocr', 'markers', 'quality')
PROFILES = {'inspection': 'quality,qr,markers,ocr', 'interaction': 'quality,detect,pose',
            'spatial': 'quality,markers,depth', 'all': ','.join(TASKS)}


def read_image(path):
    frame = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Invalid image: ' + str(path))
    return frame


def save_image(path, frame):
    ok, data = cv2.imencode(Path(path).suffix, frame)
    if not ok:
        raise RuntimeError('Image encoding failed')
    with Path(path).open('xb') as output:
        output.write(data.tobytes())


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    union = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + max(0., b[2]-b[0]) * max(0., b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


class BoxTracker:
    """Short-term, class-aware IoU IDs, not identity recognition or re-ID."""
    def __init__(self, max_gap_s=1.0):
        self.previous = []
        self.next_id = 1
        self.stamp = None
        self.max_gap_s = max_gap_s

    def update(self, boxes, stamp):
        if self.stamp is not None and (stamp <= self.stamp or stamp-self.stamp > self.max_gap_s):
            self.previous = []
        used = set()
        for box in boxes:
            candidates = [(iou(box['xyxy'], old['xyxy']), n) for n, old in enumerate(self.previous)
                          if n not in used and box['class_id'] == old['class_id']]
            score, match = max(candidates, default=(0., -1))
            if score >= .3:
                box['track_id'] = self.previous[match]['track_id']
                used.add(match)
            else:
                box['track_id'] = self.next_id
                self.next_id += 1
        self.previous = [dict(b) for b in boxes]
        self.stamp = stamp
        return boxes


class FaceEngine:
    def __init__(self, models, gallery=None, threshold=.5):
        self.detector = cv2.FaceDetectorYN_create(
            'onnx', np.frombuffer((models / 'face_detection_yunet_2023mar.onnx').read_bytes(), np.uint8),
            np.empty(0, np.uint8), (320, 320), .8, .3, 5000)
        self.recognizer = cv2.FaceRecognizerSF_create(
            'onnx', np.frombuffer((models / 'face_recognition_sface_2021dec.onnx').read_bytes(), np.uint8),
            np.empty(0, np.uint8))
        self.threshold = threshold
        self.gallery = json.loads(gallery.read_text(encoding='utf-8')) if gallery and gallery.exists() else []
        for item in self.gallery:
            feature = np.asarray(item['embedding'], np.float32)
            if feature.shape != (128,) or not np.isfinite(feature).all() or np.linalg.norm(feature) < 1e-8:
                raise ValueError('Invalid face gallery embedding')

    def detect(self, frame):
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def feature(self, frame, face):
        feature = self.recognizer.feature(self.recognizer.alignCrop(frame, face)).reshape(-1)
        return feature / max(float(np.linalg.norm(feature)), 1e-8)

    def infer(self, frame):
        annotated = frame.copy()
        rows = []
        for face in self.detect(frame):
            feature = self.feature(frame, face)
            best_name, best_score = 'unknown', None
            for item in self.gallery:
                reference = np.asarray(item['embedding'], np.float32)
                score = float(np.dot(feature, reference / np.linalg.norm(reference)))
                if best_score is None or score > best_score:
                    best_name, best_score = item['name'], score
            name = best_name if best_score is not None and best_score >= self.threshold else 'unknown'
            x, y, w, h = map(int, face[:4])
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (90, 230, 130), 2)
            cv2.putText(annotated, name, (x, max(14, y-6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 230, 130), 1)
            rows.append({'xywh': face[:4].tolist(), 'landmarks': face[4:14].reshape(5, 2).tolist(),
                         'confidence': float(face[-1]), 'identity': name, 'cosine_similarity': best_score})
        return {'faces': rows, 'gallery_size': len(self.gallery), 'match_threshold': self.threshold,
                'note': 'Local enrolled samples only; threshold needs on-site validation'}, annotated


class VisionEngine:
    def __init__(self, tasks, models, device='cpu', gallery=None):
        self.tasks, self.device = tasks, device
        self.models = {}
        self.tracker = BoxTracker()
        if any(t in tasks for t in ('detect', 'pose', 'segment')):
            from ultralytics import YOLO
            for task, filename in [('detect', 'yolov8n.pt'), ('pose', 'yolov8n-pose.pt'), ('segment', 'yolov8n-seg.pt')]:
                if task in tasks:
                    path = models / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.models[task] = YOLO(str(path))
        if 'faces' in tasks:
            self.faces = FaceEngine(models, gallery)
        if 'qr' in tasks:
            self.qr = cv2.QRCodeDetector()
        if 'ocr' in tasks:
            from .vision_extra import OcrEngine
            self.ocr = OcrEngine()
        if 'markers' in tasks:
            from .vision_extra import MarkerEngine
            self.markers = MarkerEngine()
        if 'depth' in tasks:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            path = models / 'depth-anything-v2-small'
            self.depth_processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(str(path), local_files_only=True)
            self.depth_model.to('cuda:' + device if device.isdigit() else device).eval()

    def infer(self, task, frame, stamp):
        if task == 'quality':
            from .vision_extra import image_quality
            data, image = image_quality(frame)
            return data, image, None
        if task in ('ocr', 'markers'):
            data, image = getattr(self, task).infer(frame)
            return data, image, None
        if task in self.models:
            result = self.models[task].predict(frame, device=self.device, imgsz=640, conf=.3, verbose=False)[0]
            boxes = [{'xyxy': xy, 'confidence': float(conf), 'class_id': int(cls),
                      'label': result.names[int(cls)]} for xy, conf, cls in zip(
                          result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
            if task == 'detect':
                self.tracker.update(boxes, stamp)
            data = {'boxes': boxes, 'speed_ms': result.speed}
            if task == 'pose':
                points = result.keypoints.data.cpu().tolist()
                data['keypoints_xy_conf'] = points
                # COCO: shoulders 5/6, wrists 9/10. Raised wrist is not a wave/open palm.
                data['raised_hand_candidates'] = [
                    any(p[w][2] >= .5 and p[s][2] >= .5 and p[w][1] < p[s][1] - 15
                        for w, s in ((9, 5), (10, 6))) for p in points if len(p) == 17]
            if task == 'segment':
                data['mask_polygons_xy'] = [p.tolist() for p in result.masks.xy] if result.masks else []
            annotated = result.plot()
            if task == 'detect':
                for box in boxes:
                    x, y = map(int, box['xyxy'][:2])
                    cv2.putText(annotated, 'ID ' + str(box['track_id']), (x, max(16, y+18)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 0), 2)
            return data, annotated, None
        if task == 'faces':
            data, image = self.faces.infer(frame)
            return data, image, None
        if task == 'qr':
            found, values, points, _ = self.qr.detectAndDecodeMulti(frame)
            image = frame.copy()
            rows = []
            if found:
                for value, corners in zip(values, points):
                    if not value:
                        continue
                    cv2.polylines(image, [corners.astype(np.int32)], True, (90, 230, 130), 2)
                    rows.append({'text': value, 'corners': corners.tolist()})
            # OpenCV's multi detector can miss a single perspective-skewed code
            # that the single-code detector decodes. Never substitute OCR text.
            if not rows:
                value, single_points, _ = self.qr.detectAndDecode(frame)
                if value and single_points is not None:
                    corners = np.asarray(single_points).reshape(4, 2)
                    cv2.polylines(image, [corners.astype(np.int32)], True, (90, 230, 130), 2)
                    rows.append({'text': value, 'corners': corners.tolist()})
            return {'codes': rows, 'note': 'Decoded text is data, never executed'}, image, None
        if task == 'depth':
            import torch
            inputs = self.depth_processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors='pt')
            inputs = {key: value.to(self.depth_model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.depth_model(**inputs)
                raw = torch.nn.functional.interpolate(outputs.predicted_depth.unsqueeze(1),
                    size=frame.shape[:2], mode='bicubic', align_corners=False).squeeze().float().cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError('Non-finite depth prediction')
            lo, hi = np.percentile(raw, [2, 98])
            normalized = np.clip((raw-lo) / max(float(hi-lo), 1e-6), 0, 1)
            image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            return {'units': 'relative_inverse_depth', 'metric': False,
                    'note': 'Larger values approximately nearer; not meters or obstacle clearance'}, image, raw
        raise ValueError('Unknown vision task ' + task)


def stereo_depth(left, right, calibration, left_stamp, right_stamp):
    """Rectify a calibrated, synchronized stereo pair; return metric Z + validity."""
    if calibration.get('quality_gate_passed') is False:
        raise ValueError('Calibration failed its quality gate')
    if not all(math.isfinite(t) for t in (left_stamp, right_stamp)) or abs(left_stamp-right_stamp) > .02:
        raise ValueError('Stereo pair must be synchronized within 20 ms on a shared clock')
    if left.shape != right.shape or list(left.shape[1::-1]) != calibration['image_size']:
        raise ValueError('Pair dimensions must equal calibration image_size [width,height]')
    k1, k2 = (np.asarray(calibration[k], np.float64).reshape(3, 3) for k in ('K1', 'K2'))
    d1, d2 = (np.asarray(calibration[k], np.float64) for k in ('D1', 'D2'))
    rotation = np.asarray(calibration['R'], np.float64).reshape(3, 3)
    translation = np.asarray(calibration['T_m'], np.float64).reshape(3)
    if not all(np.isfinite(a).all() for a in (k1, k2, d1, d2, rotation, translation)):
        raise ValueError('Calibration contains non-finite numbers')
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-3):
        raise ValueError('R must be a proper rotation matrix')
    if min(k1[0, 0], k1[1, 1], k2[0, 0], k2[1, 1]) <= 0 or not .001 <= np.linalg.norm(translation) <= 1.0:
        raise ValueError('Invalid focal length or baseline; T_m must be meters')
    size = tuple(calibration['image_size'])
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(k1, d1, k2, d2, size, rotation, translation,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(p2[0, 3]) <= abs(p2[1, 3]) or p2[0, 3] >= 0:
        raise ValueError('Requires horizontal stereo with left/right ordering for positive disparity')
    rectified = []
    for image, k, d, r, p in ((left, k1, d1, r1, p1), (right, k2, d2, r2, p2)):
        mx, my = cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
        rectified.append(cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mx, my, cv2.INTER_LINEAR))
    disparities = min(128, (size[0] // 4 // 16) * 16)
    if disparities < 16:
        raise ValueError('Image too narrow for stereo matching')
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=disparities, blockSize=5,
        P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=80,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = matcher.compute(*rectified).astype(np.float32) / 16
    depth = cv2.reprojectImageTo3D(disparity, q)[:, :, 2]
    x, y, w, h = cv2.getValidDisparityROI(roi1, roi2, 0, disparities, 5)
    roi_mask = np.zeros(depth.shape, bool)
    roi_mask[y:y+h, x:x+w] = True
    valid = roi_mask & (disparity > 0) & np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    return disparity, depth, valid


def report_html(output, rows):
    cards = []
    for row in rows:
        for task, result in row.get('tasks', {}).items():
            caption = html.escape(f"Frame {row['frame']} / {task} / {result['status']}")
            media = f'<img src="{html.escape(result["image"])}">' if 'image' in result else ''
            detail = html.escape(json.dumps(result, ensure_ascii=False, indent=2))
            text_result = result.get('result', {}).get('text')
            text_html = '<pre>' + html.escape(text_result) + '</pre>' if text_result else ''
            cards.append(f'<article><h2>{caption}</h2>{media}{text_html}<details><summary>结果 JSON</summary><pre>{detail}</pre></details></article>')
        if 'capture_error' in row:
            cards.append('<article><h2>取图失败</h2><pre>' + html.escape(row['capture_error']) + '</pre></article>')
    document = '''<!doctype html><meta charset="utf-8"><title>Aura 视觉工作台</title>
<style>body{background:#111923;color:#e6eff9;font:16px system-ui;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}article{padding:16px;background:#1d2939;border-radius:12px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:17px}</style>
<h1>Aura 视觉工作台</h1><p>本机推理 · 感知预览 · 深度图单位以 JSON 标注为准</p><main>'''
    if rows and all(r.get('source_kind') == 'offline_image' for r in rows):
        source_note = '离线图像推理 · 非实机采集'
    elif rows and all(r.get('source_kind') == 'controlled_image_sequence' for r in rows):
        source_note = '可控图片序列联调 · 逐帧重新推理 · 非现场采集'
    elif rows and all(r.get('source_kind') == 'mujoco_simulation' for r in rows):
        source_note = 'MuJoCo 仿真相机渲染 · 非实机采集 · 场景与相机参数并非实机标定'
    else:
        source_note = '相机/流或旧记录：是否成功采集请核对每帧来源及结果'
    event_link = '<p>' + source_note + '</p>'
    event_link += '<p><a style="color:#9cddff" href="events.html">查看事件时间线与性能摘要</a></p>' if (output/'events.html').is_file() else ''
    (output / 'report.html').write_text(document + event_link + ''.join(cards) + '</main>', encoding='utf-8')


def run(args):
    from .vision_events import EventMonitor, analyze, save_analysis, validate_zone
    tasks = (args.tasks or PROFILES.get(getattr(args, 'profile', None), 'detect,pose,segment,faces,qr')).split(',')
    zone = validate_zone(getattr(args, 'zone', None))
    event_settings = {'zone': zone, 'confirm_frames': getattr(args, 'event_confirm_frames', 2),
                      'max_gap_s': getattr(args, 'event_max_gap_s', 3.)}
    monitor = EventMonitor(**event_settings)
    if not tasks or any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('Tasks must be a unique comma-separated subset of ' + ','.join(TASKS))
    if not 1 <= args.frames <= 300 or not math.isfinite(args.fps) or not .1 <= args.fps <= 30:
        raise ValueError('frames must be 1..300; fps must be .1..30')
    if args.image and args.frames != 1:
        raise ValueError('A still image supports exactly one frame')
    if zone is not None and 'detect' not in tasks:
        raise ValueError('zone requires the detect task')
    if args.output.exists():
        raise FileExistsError(args.output)
    # One missing optional model must not disable all other selected tasks.
    engines, init_errors = {}, {}
    for task in tasks:
        try:
            engines[task] = VisionEngine([task], args.models, args.device, args.gallery)
        except Exception as exc:
            init_errors[task] = str(exc)
    args.output.mkdir(parents=True, exist_ok=False)
    client = AuraClient(args.host, 8765, 15.)
    rows = []
    session_start = None
    for index in range(args.frames):
        started = time.monotonic()
        if session_start is None:
            session_start = started
        row = {'frame': index, 'created_at': datetime.now(timezone.utc).isoformat(),
               'source_kind': getattr(args, 'source_kind', None) or ('offline_image' if args.image else 'camera_request'),
               'sample_time_s': started, 'motion_sent': False, 'tasks': {}}
        try:
            if args.image:
                frame = read_image(args.image)
                row['source'] = str(args.image)
            else:
                if args.stream_url:
                    data, capture_headers = fetch_stream_frame(args.stream_url)
                    if capture_headers.get('x-aura-source') == 'mujoco_simulation':
                        row['source_kind'] = 'mujoco_simulation'
                        row['camera_metadata'] = {key: capture_headers.get(key) for key in
                            ('x-aura-source', 'x-frame-sequence', 'x-frame-age-ms')}
                else:
                    data = client.http('/api/media/camera/snapshot')
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Camera returned invalid image')
                row['source'] = args.stream_url or client.base
            row['capture_ms'] = (time.monotonic()-started)*1000
            row['image_size'] = list(frame.shape[1::-1])
            save_image(args.output / f'{index:04d}-camera.jpg', frame)
        except Exception as exc:
            row['capture_error'] = str(exc)
            row['events'] = monitor.update(row, started-session_start)
            rows.append(row)
            write_json(args.output / f'{index:04d}.json', row)
            break
        for task in tasks:
            tick = time.monotonic()
            try:
                if task in init_errors:
                    raise RuntimeError('Initialization failed: ' + init_errors[task])
                data, annotated, array = engines[task].infer(task, frame, started)
                filename = f'{index:04d}-{task}.jpg'
                save_image(args.output / filename, annotated)
                if array is not None:
                    np.save(args.output / f'{index:04d}-{task}.npy', array, allow_pickle=False)
                row['tasks'][task] = {'status': 'pass', 'elapsed_ms': (time.monotonic()-tick)*1000,
                                      'image': filename, 'result': data}
            except Exception as exc:
                row['tasks'][task] = {'status': 'failed', 'error': str(exc)}
        row['events'] = monitor.update(row, started-session_start)
        rows.append(row)
        write_json(args.output / f'{index:04d}.json', row)
        print(json.dumps({'frame': index, 'tasks': {k: v['status'] for k, v in row['tasks'].items()},
                          'capture_ms': row['capture_ms'], 'events': row['events']}, ensure_ascii=False), flush=True)
        if index+1 < args.frames:
            time.sleep(max(0., 1/args.fps - (time.monotonic()-started)))
    write_json(args.output / 'results.json', rows)
    save_analysis(args.output, analyze(rows, **event_settings))
    report_html(args.output, rows)
    return int(any('capture_error' in row or any(t['status'] != 'pass' for t in row['tasks'].values()) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    live = sub.add_parser('run')
    source = live.add_mutually_exclusive_group()
    source.add_argument('--image', type=Path)
    source.add_argument('--stream-url')
    live.add_argument('--host', default='192.168.4.1')
    live.add_argument('--source-kind',choices=['mujoco_simulation'],help='Explicit provenance for a virtual camera/image')
    selection = live.add_mutually_exclusive_group()
    selection.add_argument('--tasks')
    selection.add_argument('--profile', choices=PROFILES)
    live.add_argument('--zone', type=float, nargs=4, help='Normalized image rectangle x1 y1 x2 y2; detect only')
    live.add_argument('--event-confirm-frames', type=int, default=2)
    live.add_argument('--event-max-gap-s', type=float, default=3.)
    live.add_argument('--models', type=Path, default=Path('models/vision'))
    live.add_argument('--device', default='cpu')
    live.add_argument('--gallery', type=Path)
    live.add_argument('--frames', type=int, default=1)
    live.add_argument('--fps', type=float, default=2.)
    live.add_argument('--output', type=Path, required=True)
    enroll = sub.add_parser('enroll', help='Register an explicitly supplied, labeled face locally')
    enroll.add_argument('--image', type=Path, required=True)
    enroll.add_argument('--name', required=True)
    enroll.add_argument('--gallery', type=Path, required=True)
    enroll.add_argument('--models', type=Path, default=Path('models/vision'))
    stereo = sub.add_parser('stereo', help='Calibrated synchronized raw pair to metric depth')
    for key in ('left', 'right', 'calibration', 'output'):
        stereo.add_argument('--' + key, type=Path, required=True)
    stereo.add_argument('--left-stamp', type=float, required=True)
    stereo.add_argument('--right-stamp', type=float, required=True)
    doctor = sub.add_parser('doctor', help='Initialize each local perception task independently; no robot access')
    doctor.add_argument('--models', type=Path, default=Path('models/vision'))
    doctor.add_argument('--device', default='cpu')
    doctor.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'doctor':
        return doctor_run(args)
    if args.command == 'run':
        return run(args)
    if args.command == 'enroll':
        if not args.name.strip() or len(args.name) > 80:
            raise ValueError('A nonempty name of at most 80 characters is required')
        engine = FaceEngine(args.models, args.gallery)
        frame = read_image(args.image)
        faces = engine.detect(frame)
        if len(faces) != 1:
            raise ValueError('Enrollment requires exactly one detected face')
        if any(item['name'] == args.name for item in engine.gallery):
            raise ValueError('Name already registered; original gallery preserved')
        engine.gallery.append({'name': args.name, 'embedding': engine.feature(frame, faces[0]).tolist()})
        args.gallery.parent.mkdir(parents=True, exist_ok=True)
        # The explicit enroll command updates this local gallery, not photos elsewhere.
        temporary = args.gallery.with_suffix('.pending.json')
        write_json(temporary, engine.gallery)
        temporary.replace(args.gallery)
        print('Registered one local face sample; no cloud upload')
        return 0
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    disparity, depth, valid = stereo_depth(read_image(args.left), read_image(args.right), calibration,
                                          args.left_stamp, args.right_stamp)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'depth_m.npy', depth, allow_pickle=False)
    np.save(args.output / 'disparity_px.npy', disparity, allow_pickle=False)
    save_image(args.output / 'valid.png', valid.astype(np.uint8)*255)
    write_json(args.output / 'result.json', {'units': 'meters', 'valid_ratio': float(valid.mean()),
        'baseline_m': float(np.linalg.norm(calibration['T_m'])), 'motion_sent': False,
        'note': 'Accuracy depends on calibration, synchronization and matching; not navigation clearance'})
    return 0 if valid.any() else 1


def doctor_run(args):
    rows = {}
    for task in TASKS:
        started = time.monotonic()
        try:
            engine = VisionEngine([task], args.models, args.device)
            result, _, _ = engine.infer(task, np.zeros((480, 640, 3), np.uint8), started)
            rows[task] = {'status': 'smoke_pass', 'elapsed_ms': (time.monotonic()-started)*1000}
            del engine, result
        except Exception as exc:
            rows[task] = {'status': 'failed', 'error': str(exc)}
    write_json(args.output, {'tasks': rows, 'robot_contacted': False, 'motion_sent': False,
        'note': 'Synthetic blank-frame smoke tests only, not live accuracy or hardware health.',
        'not_validated_here': ['metric_stereo_calibration_and_sync', 'SLAM_Nav2',
                               'battery_percentage_and_charging', 'microphone_audio', 'cloud_ASR']})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return int(any(row['status'] != 'smoke_pass' for row in rows.values()))


if __name__ == '__main__':
    raise SystemExit(main())
