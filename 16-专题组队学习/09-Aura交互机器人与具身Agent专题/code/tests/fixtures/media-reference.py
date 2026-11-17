from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import math
import mimetypes
from pathlib import Path
import re
from typing import Any

from PIL import Image




IMAGE_FIELD_ORDER = {
    "rgb_raw_path": 0,
    # The BEHAVIOR v2 capture contract uses rgb_path for the untouched frame.
    "rgb_path": 0,
    "image_url": 10,
    "rgb_overlay_path": 20,
    "marked_image_url": 30,
    # rgb_main* is the display image and may contain a path/grasp overlay.
    "rgb_main_path": 40,
    "rgb_main": 50,
    "rgb": 60,
}
IMAGE_FIELD_ROLES = {
    "rgb_raw_path": "raw_rgb",
    "rgb_path": "raw_rgb",
    "image_url": "raw_rgb",
    "rgb_overlay_path": "auxiliary_overlay",
    "marked_image_url": "marked_overlay",
    "rgb_main_path": "raw_rgb",
    "rgb_main": "display_rgb",
    "rgb": "display_rgb",
}
IMAGE_FIELDS = frozenset(IMAGE_FIELD_ORDER)
DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$"
)
EMBEDDED_DATA_URL_RE = re.compile(
    r"data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,[A-Za-z0-9+/=\s]+"
)

RAW_MEDIA_ORDER = (
    "rgb_raw_path",
    "rgb_path",
    "image_url",
    "rgb_main_path",
    "rgb_main",
    "rgb",
    "rgb_overlay_path",
    "marked_image_url",
)
PLANNING_MEDIA_ORDER = (
    "marked_image_url",
    "rgb_overlay_path",
    "rgb_main",
    "rgb",
    "rgb_main_path",
    "rgb_path",
    "image_url",
    "rgb_raw_path",
)
ANNOTATED_MEDIA_ORDER = (
    "rgb_overlay_path",
    "rgb_main",
    "rgb",
    "rgb_main_path",
    "marked_image_url",
    "rgb_path",
    "image_url",
    "rgb_raw_path",
)
PLANNING_OVERLAY_TOOLS = frozenset(
    {"plan_grasp_point_filter_rgbd_lite", "plan_press_point"}
)
RAW_GROUNDING_TOOLS = frozenset(
    {
        "set_arm_to_grasp_position",
        "adjust_left_eef_pose_in_head_frame",
        "adjust_right_eef_pose_in_head_frame",
    }
)
WRIST_OVERLAY_TOOLS = frozenset(
    {
        "capture_left_wrist_camera",
        "capture_right_wrist_camera",
        "adjust_left_eef_pose_in_wrist_frame",
        "adjust_right_eef_pose_in_wrist_frame",
        "exec_plan_pose",
    }
)
ROLE_OVERRIDE_LABELS = {
    "path_overlay": frozenset({"rgb_overlay_path", "rgb_main", "rgb"}),
    "grasp_volume_overlay": frozenset(
        {"rgb_overlay_path", "rgb_main", "rgb"}
    ),
    "planning_overlay": frozenset(
        {"marked_image_url", "rgb_overlay_path", "rgb_main", "rgb"}
    ),
}


@dataclass(frozen=True)
class Media:
    label: str
    role: str
    mime_type: str
    data: bytes
    sha256: str
    width: int | None = None
    height: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    source_image_id: str = ""
    source_feed: str = ""

    @classmethod
    def create(
        cls,
        *,
        label: str,
        mime_type: str,
        data: bytes,
        role: str = "",
        source_size: tuple[int | None, int | None] | None = None,
        source_image_id: str = "",
        source_feed: str = "",
    ) -> "Media":
        normalized_label = label.lower()
        width, height = _image_dimensions(data)
        return cls(
            label=normalized_label,
            role=role or IMAGE_FIELD_ROLES.get(normalized_label, "image"),
            mime_type=mime_type,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
            source_width=source_size[0] if source_size else width,
            source_height=source_size[1] if source_size else height,
            source_image_id=source_image_id,
            source_feed=source_feed,
        )

    def with_role(self, role: str) -> "Media":
        return replace(self, role=role)

    def public(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "bytes": len(self.data),
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


@dataclass
class MediaSelection:
    rest_candidate_count: int
    selected: list[Media]
    warnings: list[str]


def _detect_mime(data: bytes, hint: str = "") -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return (
            int.from_bytes(data[6:8], "little"),
            int.from_bytes(data[8:10], "little"),
        )
    if data.startswith(b"\xff\xd8\xff"):
        offset = 2
        start_of_frame = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while offset + 4 <= len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(data):
                break
            segment_length = int.from_bytes(data[offset : offset + 2], "big")
            if segment_length < 2 or offset + segment_length > len(data):
                break
            if marker in start_of_frame and segment_length >= 7:
                return (
                    int.from_bytes(data[offset + 5 : offset + 7], "big"),
                    int.from_bytes(data[offset + 3 : offset + 5], "big"),
                )
            offset += segment_length
    return None, None


def _nested_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _nested_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _nested_value(value, key)
            if found is not None:
                return found
    return None


def _selection_policy(tool_name: str, payload: Any) -> tuple[str, tuple[str, ...]]:
    if not tool_name:
        return "raw_rgb", RAW_MEDIA_ORDER

    result_ok = _nested_value(payload, "ok")
    plan_preview = (
        _nested_value(payload, "marked_image_url")
        or _nested_value(payload, "plan_preview_overlay")
    )
    gripper_visualization = _nested_value(payload, "gripper_visualization")
    has_plan_preview = bool(plan_preview) or (
        isinstance(gripper_visualization, dict)
        and gripper_visualization.get("ok") is True
    )
    if (
        tool_name in PLANNING_OVERLAY_TOOLS
        and result_ok is not False
        and has_plan_preview
    ):
        return "planning_overlay", PLANNING_MEDIA_ORDER

    feed = str(_nested_value(payload, "feed") or "").lower()
    if tool_name in RAW_GROUNDING_TOOLS and feed in {"", "head", "main"}:
        return "raw_rgb", RAW_MEDIA_ORDER
    if feed in {"head", "main"}:
        return "path_overlay", ANNOTATED_MEDIA_ORDER
    grasp_overlay = _nested_value(payload, "grasp_zone_overlay")
    if (
        tool_name in WRIST_OVERLAY_TOOLS
        or "wrist" in feed
        or (isinstance(grasp_overlay, dict) and grasp_overlay.get("ok") is True)
    ):
        return "grasp_volume_overlay", ANNOTATED_MEDIA_ORDER
    return "path_overlay", ANNOTATED_MEDIA_ORDER


class MediaExtractor:
    def __init__(
        self,
        client: Any,
        *,
        max_bytes: int,
        max_images: int,
        max_model_bytes: int = 256 * 1024,
        max_model_edge: int = 720,
        jpeg_quality: int = 95,
    ) -> None:
        self.client = client
        self.max_bytes = max_bytes
        # Preserve constructor compatibility while enforcing one model-visible image.
        self.max_images = min(max_images, 1)
        self.max_model_bytes = max_model_bytes
        self.max_model_edge = max_model_edge
        self.jpeg_quality = jpeg_quality

    def extract(self, payload: Any, *, tool_name: str = "") -> MediaSelection:
        references: list[tuple[int, str, str, str, str, str]] = []

        def walk(node: Any, key: str = "", path: str = "$", image_id: str = "", feed: str = "") -> None:
            if isinstance(node, dict):
                image_id = str(node.get("image_id") or image_id)
                feed = str(node.get("feed") or feed)
                for child_key, value in node.items():
                    child = str(child_key)
                    if child in {"replay_frames", "history", "diagnostic_payload"}:
                        continue
                    walk(value, child, f"{path}.{child}", image_id, feed)
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, key, f"{path}[{index:08d}]", image_id, feed)
            elif key.lower() in IMAGE_FIELDS and isinstance(node, str) and node:
                label = key.lower()
                references.append(
                    (IMAGE_FIELD_ORDER[label], path, label, node, image_id, feed)
                )

        walk(payload)
        references.sort(key=lambda item: (item[0], item[1], item[3]))
        loaded: list[tuple[str, Media]] = []
        warnings: list[str] = []
        seen_references: set[tuple[str, str]] = set()
        expected_id = str(payload.get("image_id") or "") if isinstance(payload, dict) else ""
        if not expected_id and isinstance(payload, dict) and isinstance(payload.get("observation"), dict):
            expected_id = str(payload["observation"].get("image_id") or "")
        for _, path, label, reference, image_id, feed in references:
            if expected_id and image_id and expected_id != image_id:
                warnings.append(f"Ignored {label} from non-current image_id={image_id}.")
                continue
            reference_key = (label, reference)
            if reference_key in seen_references:
                continue
            seen_references.add(reference_key)
            try:
                data, hint = self._load(reference)
                mime = _detect_mime(data, hint)
                if mime is None:
                    raise ValueError("reference did not contain a recognized image")
                media = Media.create(
                    label=label,
                    role=IMAGE_FIELD_ROLES[label],
                    mime_type=mime,
                    data=data,
                    source_image_id=image_id,
                    source_feed=feed,
                )
                loaded.append((path, media))
            except Exception as exc:
                warnings.append(f"Unable to load {label}: {exc}")

        role, label_order = _selection_policy(tool_name, payload)
        rank = {label: index for index, label in enumerate(label_order)}
        loaded.sort(
            key=lambda item: (
                rank.get(item[1].label, len(rank)),
                item[0],
                item[1].sha256,
            )
        )
        selected: list[Media] = []
        if loaded and self.max_images:
            media = loaded[0][1]
            if media.label in ROLE_OVERRIDE_LABELS.get(role, frozenset()):
                media = media.with_role(role)
            try:
                media = self._bound_for_model(media)
            except Exception as exc:
                warnings.append(
                    f"Selected {media.label} image omitted: unable to enforce "
                    f"the {self.max_model_bytes}-byte model limit ({exc})"
                )
            else:
                selected.append(media)
        return MediaSelection(
            rest_candidate_count=len({media.sha256 for _, media in loaded}),
            selected=selected,
            warnings=warnings,
        )


    def _bound_for_model(self, media: Media) -> Media:
        dimensions_fit = (
            media.width is not None
            and media.height is not None
            and media.width <= self.max_model_edge
            and media.height <= self.max_model_edge
        )
        if len(media.data) <= self.max_model_bytes and dimensions_fit:
            return media

        quality_floor = min(self.jpeg_quality, 90)
        attempts = (
            self.jpeg_quality,
            max(quality_floor, self.jpeg_quality - 2),
            quality_floor,
        )
        seen_attempts: set[int] = set()
        last_error = "converter produced no bounded image"
        for quality in attempts:
            if quality in seen_attempts:
                continue
            seen_attempts.add(quality)
            bounded, error = self._convert_to_jpeg(
                media,
                edge=self.max_model_edge,
                quality=quality,
            )
            if error:
                last_error = error
                continue
            if (
                bounded is not None
                and len(bounded.data) <= self.max_model_bytes
                and bounded.width is not None
                and bounded.height is not None
                and bounded.width <= self.max_model_edge
                and bounded.height <= self.max_model_edge
            ):
                return bounded
            if bounded is not None:
                last_error = (
                    f"converted image was {len(bounded.data)} bytes at "
                    f"{self.max_model_edge}px quality {quality}"
                )
        raise ValueError(last_error)

    def _convert_to_jpeg(
        self,
        media: Media,
        *,
        edge: int,
        quality: int,
    ) -> tuple[Media | None, str]:
        try:
            # Pillow and its image codecs are in this workspace's environment.
            # Keep frozen RGB aligned with depth: do not apply EXIF orientation,
            # crop or enlarge. A native 720 x 720 head view keeps its pixels.
            with Image.open(BytesIO(media.data)) as source:
                source.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                converted = source.convert("RGB")
            converted.info.clear()
            output = BytesIO()
            converted.save(output, format="JPEG", quality=quality,
                           subsampling=0, progressive=True)
            data = output.getvalue()
        except (OSError, ValueError) as exc:
            return None, str(exc)
        return (
            Media.create(
                label=media.label,
                role=media.role,
                mime_type="image/jpeg",
                data=data,
                source_size=(media.source_width, media.source_height),
                source_image_id=media.source_image_id,
                source_feed=media.source_feed,
            ),
            "",
        )

    def _load(self, reference: str) -> tuple[bytes, str]:
        match = DATA_URL_RE.fullmatch(reference)
        if match:
            encoded = "".join(match.group("data").split())
            data = base64.b64decode(encoded, validate=True)
            if len(data) > self.max_bytes:
                raise ValueError(f"inline image exceeds {self.max_bytes} bytes")
            return data, match.group("mime")

        path = Path(reference).expanduser()
        if path.is_absolute() and path.is_file():
            with path.open("rb") as stream:
                data = stream.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise ValueError(f"local image exceeds {self.max_bytes} bytes")
            return data, mimetypes.guess_type(path.name)[0] or ""

        if reference.startswith(("http://", "https://", "/")):
            return self.client.get_media(reference, self.max_bytes)
        raise ValueError("unsupported image reference")


def safe_json(value: Any, key: str = "") -> Any:
    """Remove inline media and non-finite values from model/recording JSON."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): safe_json(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item, key) for item in value]
    if isinstance(value, str):
        if DATA_URL_RE.match(value):
            mime = value[5:].split(";", 1)[0]
            return {"omitted": "inline_image", "mime_type": mime}
        if key.lower() in IMAGE_FIELDS:
            return {"media_reference": key.lower()}
        if EMBEDDED_DATA_URL_RE.search(value):
            return EMBEDDED_DATA_URL_RE.sub(
                lambda match: "[inline image omitted: " + match.group("mime") + "]",
                value,
            )
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)
