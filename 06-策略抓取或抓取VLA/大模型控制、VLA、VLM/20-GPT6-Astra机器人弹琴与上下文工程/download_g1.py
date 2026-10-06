"""Download only the pinned G1 model and its referenced meshes, not Menagerie history."""

import argparse
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/unitree_g1/"


def download(destination, proxy=None, use_aria2=False):
    destination.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy} if proxy else {}))
    for name in ("g1.xml", "LICENSE", "README.md"):
        with opener.open(BASE + name, timeout=60) as response:
            content = response.read()
        (destination / name).write_bytes(content)
    root = ET.fromstring((destination / "g1.xml").read_bytes())
    files = sorted({"assets/" + mesh.attrib["file"] for mesh in root.findall("./asset/mesh")})
    for name in files:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unexpected model asset path")
    (destination / "assets").mkdir(exist_ok=True)
    if use_aria2:
        urls = "".join(BASE + urllib.parse.quote(name) + "\n  dir=" + str(destination / "assets") + "\n  out=" + Path(name).name + "\n" for name in files)
        listing = destination / "download-urls.txt"
        listing.write_text(urls, encoding="utf-8")
        command = ["aria2c", "-c", "-j", "6", "-x", "4", "-s", "4", "--allow-overwrite=false", "--auto-file-renaming=false", "--console-log-level=warn", "--input-file=" + str(listing)]
        if proxy:
            command += ["--all-proxy=" + proxy, "--http-proxy=" + proxy, "--https-proxy=" + proxy]
        subprocess.run(command, check=True)
    else:
        for name in files:
            with opener.open(BASE + urllib.parse.quote(name), timeout=90) as response:
                (destination / name).write_bytes(response.read())
    records = {}
    for name in ["g1.xml", "LICENSE", "README.md"] + files:
        data = (destination / name).read_bytes()
        if not data:
            raise ValueError("Empty model resource: " + name)
        records[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"repository": "google-deepmind/mujoco_menagerie", "commit": COMMIT, "model": "unitree_g1/g1.xml", "license": "BSD-3-Clause", "files": records}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--aria2", action="store_true")
    args = parser.parse_args()
    manifest = download(args.out.resolve(), args.proxy, args.aria2)
    print(json.dumps({"commit": COMMIT, "files": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"].values())}))


if __name__ == "__main__":
    main()
