"""Unit tests for wire_payload_guard_candidate."""

import base64
import io
import unittest
from PIL import Image

from wire_payload_guard import (
    GATEWAY_HARD_LIMIT_BYTES,
    SAFE_PAYLOAD_CEILING_BYTES,
    PayloadLimitExceededError,
    transcode_image_to_jpeg,
    inspect_turn_inputs,
    optimize_frame_inputs_if_needed,
    assert_inputs_within_safe_ceiling,
)


class WirePayloadGuardCandidateTests(unittest.TestCase):
    def test_transcode_image_to_jpeg_dimensions_and_format(self):
        # Create a textured image representative of natural camera frames
        img = Image.effect_noise((720, 720), 50).convert('RGB')
        stream = io.BytesIO()
        img.save(stream, format='PNG')
        png_bytes = stream.getvalue()

        jpeg_bytes, w, h = transcode_image_to_jpeg(png_bytes, quality=85)
        self.assertEqual((w, h), (720, 720))
        self.assertLess(len(jpeg_bytes), len(png_bytes))

        # Verify decoded format
        with Image.open(io.BytesIO(jpeg_bytes)) as decoded:
            self.assertEqual(decoded.format, 'JPEG')
            self.assertEqual(decoded.size, (720, 720))

    def test_inspect_turn_inputs_detects_ceiling_and_hard_limit(self):
        small_items = [{'type': 'text', 'text': 'short instruction'}]
        report = inspect_turn_inputs(small_items)
        self.assertFalse(report['exceeds_safe_ceiling'])
        self.assertFalse(report['exceeds_hard_gateway_limit'])
        self.assertEqual(report['text_count'], 1)
        self.assertEqual(report['images_count'], 0)

        # Huge mock payload exceeding 10 MiB
        huge_text = 'A' * (GATEWAY_HARD_LIMIT_BYTES + 1024)
        large_items = [{'type': 'text', 'text': huge_text}]
        report_large = inspect_turn_inputs(large_items)
        self.assertTrue(report_large['exceeds_safe_ceiling'])
        self.assertTrue(report_large['exceeds_hard_gateway_limit'])

    def test_optimize_frame_inputs_transcodes_png_to_jpeg(self):
        img = Image.effect_noise((480, 480), 50).convert('RGB')
        stream = io.BytesIO()
        img.save(stream, format='PNG')
        png_bytes = stream.getvalue()
        png_url = 'data:image/png;base64,' + base64.b64encode(png_bytes).decode('ascii')

        raw_inputs = [
            {'type': 'text', 'text': 'head camera perception'},
            {'type': 'image', 'url': png_url}
        ]

        optimized, count = optimize_frame_inputs_if_needed(raw_inputs, quality=85)
        self.assertEqual(count, 1)
        self.assertEqual(len(optimized), 2)
        self.assertTrue(optimized[1]['url'].startswith('data:image/jpeg;base64,'))
        self.assertLess(len(optimized[1]['url']), len(png_url))

    def test_assert_inputs_within_safe_ceiling(self):
        safe_items = [{'type': 'text', 'text': 'safe message'}]
        inspection = assert_inputs_within_safe_ceiling(safe_items)
        self.assertFalse(inspection['exceeds_safe_ceiling'])

        dangerous_items = [{'type': 'text', 'text': 'X' * (SAFE_PAYLOAD_CEILING_BYTES + 10)}]
        with self.assertRaises(PayloadLimitExceededError):
            assert_inputs_within_safe_ceiling(dangerous_items)


if __name__ == '__main__':
    unittest.main()
