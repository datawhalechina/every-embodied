"""Convert the pinned GalaxeaManipSim R1 Pro URDF to a teaching MJCF model."""

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

COMMIT = "abe7f5161eeaa150e6eaffdf443af5df7f23f356"


def prepare(repo, output, keep_grippers=False):
    import mujoco

    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Use GalaxeaManipSim commit " + COMMIT)
    source = repo / "galaxea_sim/assets/r1_pro/new_robot.urdf"
    root = ET.fromstring(source.read_bytes())
    # Remove the parallel gripper subtrees; the arm frames/axes remain unchanged.
    removed = set() if keep_grippers else {"left_gripper_link", "right_gripper_link"}
    while True:
        children = {j.find("child").get("link") for j in root.findall("joint") if j.find("parent").get("link") in removed}
        if children <= removed:
            break
        removed |= children
    for element in list(root):
        if (element.tag == "link" and element.get("name") in removed) or (element.tag == "joint" and element.find("child").get("link") in removed):
            root.remove(element)
    meshes = {}
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for i, element in enumerate(link.findall(kind)):
                element.set("name", f"{link.get('name')}_{kind}_{i}")
                mesh = element.find("geometry/mesh")
                if mesh is not None:
                    path = (source.parent / mesh.get("filename")).resolve()
                    meshes[path.name] = path
                    mesh.set("filename", path.name)
    output.mkdir(parents=True, exist_ok=False)
    for name, path in meshes.items():
        shutil.copyfile(path, output / name)
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", meshdir=str(output.resolve()), discardvisual="false", fusestatic="false", balanceinertia="true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    mujoco.mj_saveLastXML(str(output / "r1_pro.xml"), model)
    xml = ET.fromstring((output / "r1_pro.xml").read_bytes())
    xml.find("compiler").set("meshdir", ".")
    for mesh in xml.findall("./asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    (output / "r1_pro.xml").write_text(ET.tostring(xml, encoding="unicode") + "\n", encoding="utf-8")
    shutil.copyfile(repo / "LICENSE", output / "LICENSE")
    shutil.copyfile(repo / "NOTICE", output / "NOTICE")
    manifest = {"repository": "https://github.com/OpenGalaxea/GalaxeaManipSim", "commit": COMMIT, "source": str(source.relative_to(repo)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "removed_links": sorted(removed), "base": "fixed_chassis", "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(output), "meshes": len(meshes), "source_commit": COMMIT}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-grippers", action="store_true")
    args = parser.parse_args()
    prepare(args.repo, args.out, keep_grippers=args.keep_grippers)
