"""Guard against HTTP wire payload exceeding reverse proxy and API gateway body thresholds.

Empirical evidence from thread 01a192a4-1d8a-7610-b518-dfb44b9076ee (log 14441)
established that compressed HTTP POST requests exceeding 10.0 MiB (10,485,760 bytes)
are rejected by the reverse proxy gateway with HTTP 400 '{"detail":"Bad Request"}'.
This module provides bounded request size guards and image transcoding utilities.
"""

import base64
import io
import json
from PIL import Image

GATEWAY_HARD_LIMIT_BYTES = 10485760  # 10.0 MiB
SAFE_PAYLOAD_CEILING_BYTES = 8912896  # 8.5 MiB


class PayloadLimitExceededError(RuntimeError):
    """Raised when an outbound model turn request exceeds gateway payload safety margins."""
    pass


def transcode_image_to_jpeg(image_bytes, quality=85):
    """Convert an image (e.g. PNG) into optimized JPEG format while preserving resolution."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb_img = img.convert('RGB')
        buf = io.BytesIO()
        rgb_img.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue(), img.width, img.height


def inspect_turn_inputs(input_items):
    """Estimate the payload size and inspect items without storing raw content."""
    total_bytes = 0
    images_count = 0
    text_count = 0
    other_count = 0

    for item in input_items:
        kind = item.get('type')
        if kind == 'text':
            text_count += 1
            total_bytes += len(item.get('text', '').encode('utf-8'))
        elif kind == 'image':
            images_count += 1
            url = item.get('url', '')
            total_bytes += len(url.encode('utf-8'))
        else:
            other_count += 1
            total_bytes += len(json.dumps(item).encode('utf-8'))

    return {
        'total_estimated_bytes': total_bytes,
        'images_count': images_count,
        'text_count': text_count,
        'other_count': other_count,
        'exceeds_safe_ceiling': total_bytes > SAFE_PAYLOAD_CEILING_BYTES,
        'exceeds_hard_gateway_limit': total_bytes > GATEWAY_HARD_LIMIT_BYTES,
    }


def optimize_frame_inputs_if_needed(raw_inputs, quality=85):
    """Transcode PNG data URLs to JPEG in turn inputs to reduce payload size."""
    optimized = []
    transcoded_count = 0

    for item in raw_inputs:
        if item.get('type') == 'image' and item.get('url', '').startswith('data:image/png;base64,'):
            prefix = 'data:image/png;base64,'
            raw_base64 = item['url'][len(prefix):]
            try:
                raw_bytes = base64.b64decode(raw_base64)
                jpeg_bytes, _, _ = transcode_image_to_jpeg(raw_bytes, quality=quality)
                # Only use JPEG if it is genuinely smaller than the PNG
                if len(jpeg_bytes) < len(raw_bytes):
                    new_url = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode('ascii')
                    optimized.append({'type': 'image', 'url': new_url})
                    transcoded_count += 1
                    continue
            except Exception:
                pass
        optimized.append(item)

    return optimized, transcoded_count


def assert_inputs_within_safe_ceiling(input_items, ceiling=SAFE_PAYLOAD_CEILING_BYTES):
    """Enforce that serialized request inputs strictly remain within safe payload margins."""
    inspection = inspect_turn_inputs(input_items)
    if inspection['exceeds_safe_ceiling']:
        raise PayloadLimitExceededError(
            f"Turn input payload estimated at {inspection['total_estimated_bytes']} bytes, "
            f"exceeding safe ceiling of {ceiling} bytes (hard gateway limit: {GATEWAY_HARD_LIMIT_BYTES} bytes)"
        )
    return inspection
