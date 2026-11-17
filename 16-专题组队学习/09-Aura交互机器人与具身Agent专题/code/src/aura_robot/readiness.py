"""Read-only readiness, result interpretation and allowlisted diagnostic export."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


REQUIREMENTS = {
    'detect': (['cv2', 'numpy', 'ultralytics', 'torch'], ['yolov8n.pt']),
    'pose': (['cv2', 'numpy', 'ultralytics', 'torch'], ['yolov8n-pose.pt']),
    'segment': (['cv2', 'numpy', 'ultralytics', 'torch'], ['yolov8n-seg.pt']),
    'faces': (['cv2', 'numpy'], ['face_detection_yunet_2023mar.onnx', 'face_recognition_sface_2021dec.onnx']),
    'qr': (['cv2', 'numpy'], []), 'markers': (['cv2', 'numpy'], []),
    'quality': (['cv2', 'numpy'], []),
    'ocr': (['cv2', 'numpy', 'rapidocr_onnxruntime', 'onnxruntime'], []),
    'depth': (['cv2', 'numpy', 'transformers', 'torch'],
              ['depth-anything-v2-small/'+name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')])}


def inventory(models):
    rows = {}
    for task, (modules, files) in REQUIREMENTS.items():
        missing_modules = [m for m in modules if importlib.util.find_spec(m) is None]
        missing_files = [name for name in files if not (models/name).is_file() or (models/name).stat().st_size == 0]
        rows[task] = {'status': 'missing' if missing_modules or missing_files else 'present_not_loaded',
                      'missing_modules': missing_modules, 'missing_files': missing_files}
    return {'tasks': rows, 'robot_contacted': False,
            'note': '只检查依赖可发现、指定文件非空；不加载模型，不检查 GPU/算子兼容或权重完整性。OCR 随包模型未单独验收。需要实际运行 doctor 或任务确认。'}


def check_sdk(host):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    def check(name):
        started = time.monotonic()
        request = urllib.request.Request(f'http://{host}:8765/api/{name}/status', method='GET')
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=2.) as response:
                body = response.read(256*1024+1)
                if len(body) > 256*1024 or not isinstance(json.loads(body), dict):
                    raise ValueError('Unexpected response')
            result = {'status': 'responded', 'advice': '状态接口返回 JSON；不代表相机、声音、电池或运动已验收。'}
        except urllib.error.HTTPError as exc:
            result = {'status': 'http_error', 'http_code': exc.code,
                      'advice': '检查 AuraOS 状态接口或授权；不自动修改凭据。'}
        except (urllib.error.URLError, TimeoutError, OSError):
            result = {'status': 'unreachable', 'advice': '检查机器人 IP、同网段连通性和 AuraOS 服务。屏幕亮不等于 SDK 在线。'}
        except (ValueError, UnicodeError):
            result = {'status': 'invalid_response', 'advice': '接口响应格式不符合预期，请核对固件/API 版本。'}
        result['elapsed_ms'] = round((time.monotonic()-started)*1000, 1)
        return name, result
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = dict(pool.map(check, ('daemon', 'media', 'motion')))
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': checks,
            'camera_captured': False, 'motion_sent': False,
            'note': '仅请求三个 /status GET 接口，不采集相机、不读 MCU 电量、不重启、不改网络。结果只是检查时刻的通信情况。'}


def failure_advice(error):
    value = str(error).lower()
    if any(word in value for word in ('no module named', 'no such file', 'filenotfound', 'initialization failed')):
        return '依赖或模型初始化未通过；先看功能准备情况，再运行 doctor，保留原始错误核对。'
    if any(word in value for word in ('timed out', 'timeout', 'unreachable', 'refused', 'urlopen')):
        return '通信/取图未完成；检查 SDK 通信与相机发布服务，不据此判断机器人关机。'
    if any(word in value for word in ('cuda', 'out of memory')):
        return '检查 GPU、显存和运行环境；可减少任务或在新控制台实例中选 CPU，不自动安装或杀其他进程。'
    return '查看原始任务日志定位；此处不把未知失败推断成硬件故障。'


def explain(row):
    if not isinstance(row, dict):
        return {'pending': {'state': 'unknown', 'message': '尚无完整帧，可能仍在加载或取图；请查看任务状态和日志。'}}
    results = {}
    if 'capture_error' in row:
        results['capture'] = {'state': 'failed', 'message': failure_advice(row['capture_error'])}
    fields = {'detect': 'boxes', 'pose': 'keypoints_xy_conf', 'segment': 'mask_polygons_xy',
              'faces': 'faces', 'qr': 'codes', 'ocr': 'lines', 'markers': 'markers'}
    for task, entry in row.get('tasks', {}).items():
        if task not in REQUIREMENTS:
            continue
        if entry.get('status') != 'pass':
            results[task] = {'state': 'failed', 'message': failure_advice(entry.get('error', ''))}
            continue
        data = entry.get('result', {})
        if task == 'quality':
            warnings = data.get('warnings')
            known = {'underexposed': '欠曝', 'overexposed': '过曝', 'low_detail_or_blur': '低细节或可能模糊'}
            if not isinstance(warnings, list):
                results[task] = {'state': 'unknown', 'message': '调用完成，但缺少画质指标。'}
            elif warnings:
                label = '、'.join(known.get(w, '未分类画质提示') for w in warnings)
                results[task] = {'state': 'warning', 'message': label+'。先检查照明/视野，不能用算法通过代替画质验收。'}
            else:
                results[task] = {'state': 'no_warning', 'message': '未触发画质启发式警告，不保证清晰或未遮挡。'}
        elif task == 'depth':
            results[task] = {'state': 'preview_only', 'message': '单目相对深度预览，不是米制距离，不能直接用于安全避障。'}
        else:
            values = data.get(fields[task])
            if not isinstance(values, list):
                results[task] = {'state': 'unknown', 'message': '调用完成但结果字段缺失，未作为有效检出。'}
            else:
                count = len(values)
                results[task] = {'state': 'detected' if count else 'empty', 'count': count,
                    'message': f'本帧输出 {count} 项；需人工核对误检、漏检。' if count else '调用完成，未检出目标；可能没有目标，也可能受光照/视角影响，不等于识别能力已验收。'}
    return results


def diagnostic(metadata, row, summary):
    # Build from a strict allowlist, never redact arbitrary logs with a regex.
    tasks = {}
    for name, value in explain(row).items():
        entry = {'state': value['state'], 'message': value['message']}
        if 'count' in value:
            entry['count'] = value['count']
        tasks[name] = entry
    timings = {}
    for task, data in (summary or {}).get('tasks', {}).items():
        if task not in REQUIREMENTS:
            continue
        timings[task] = {key: value for key, value in data.get('timing', {}).items()
                         if key in ('samples', 'p50_ms', 'p95_ms', 'max_ms')
                         and type(value) in (int, float) and math.isfinite(value) and value >= 0}
    state = metadata.get('status')
    return {'schema': 'aura-diagnostic-v1',
            'evidence_source': metadata.get('source') if metadata.get('source') in ('robot', 'sample_person', 'sample_text') else 'unknown_legacy',
            'run_status': state if state in ('starting', 'running', 'completed', 'failed', 'stopped', 'timed_out', 'unknown_after_restart') else 'unknown',
            'latest_frame_assessment': tasks, 'timing': timings,
            'privacy': '不含图像、OCR/二维码文字、人脸身份/特征、坐标、路标编号、机器人 IP、本机路径、日志或原始错误。仍含汇总数量与性能，分享前请检查。',
            'motion_sent': False}
