from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import Robot
from .mock import MockBackend
from .usb_probe import added_devices, load_snapshot, save_snapshot


CAPABILITY_AUDIT = {
    "product_family": "FrontierX Aura spherical companion robot",
    "identity_status": "FrontierX Aura; live device tree confirms D-Robotics RDK X5 V1.0",
    "compute_options": ["D-Robotics X5", "NVIDIA Orin Nano"],
    "public_sdk": "AuraOS Skill SDK is public and cloned under vendor/AuraOS-skill-store",
    "robot_runtime": "AuraOS Daemon plus ROS 2/TogetheROS Humble on Aura/RDK target",
    "legacy_documented_topics_not_all_live_verified": [
        "/cmd_vel",
        "/imu/data",
        "/battery_state",
        "/motor/connected",
        "/ai/state",
        "/ai/emotion",
        "audio_data",
        "audio_playback",
    ],
    "external_connection": "Wi-Fi/AP hotspot plus SSH, or AuraOS Desktop to Aura Daemon",
    "usb_c_role": "not documented; do not confuse external Type-C with internal /dev/ttyUSB0 motor link",
    "low_level_source": "device AuraOS source tree; see vendor documentation",
    "live_interfaces": ["http://192.168.4.1:8765/api", "ws://192.168.4.1:8765/ws/sdk"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aura-robot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="print the evidence-backed capability summary")
    subparsers.add_parser("demo", help="run a safe mock behavior")
    snapshot_parser = subparsers.add_parser("usb-snapshot", help="save present USB PnP devices")
    snapshot_parser.add_argument("path", type=Path)
    diff_parser = subparsers.add_parser("usb-diff", help="show devices added between snapshots")
    diff_parser.add_argument("before", type=Path)
    diff_parser.add_argument("after", type=Path)
    return parser


def run_demo() -> None:
    backend = MockBackend()
    with Robot(backend) as robot:
        robot.show_expression("curious")
        robot.speak("Hello from the mock backend")
        robot.move(linear=0.2, angular=0.0, duration_s=0.5)
    print("\n".join(backend.events))


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.command == "info":
        print(json.dumps(CAPABILITY_AUDIT, ensure_ascii=False, indent=2))
    elif args.command == "demo":
        run_demo()
    elif args.command == "usb-snapshot":
        count = save_snapshot(args.path)
        print(f"Saved {count} USB devices to {args.path.resolve()}")
    elif args.command == "usb-diff":
        devices = added_devices(load_snapshot(args.before), load_snapshot(args.after))
        print(json.dumps(devices, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
