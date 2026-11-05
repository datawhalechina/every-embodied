"""Record an existing DribbleAMP policy with CC0 visual-only stadium assets."""
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

ROOT = Path(__file__).resolve().parent / "scene_assets"
MIMICKIT = os.environ["DRIBBLEAMP_MIMICKIT"]
os.chdir(MIMICKIT)
sys.path.insert(0, os.path.join(MIMICKIT, "mimickit"))

import engines.newton_recorder as recorder


def load_parts(path, selector=None):
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        if selector and not selector(node):
            continue
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        xyz = trimesh.transform_points(geom.vertices, transform)
        xyz = np.column_stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1]))
        color = np.asarray(geom.visual.material.baseColorFactor[:3], dtype=np.float32) / 255
        parts.append((node, xyz, np.asarray(geom.faces, dtype=np.int32).reshape(-1), color))
    return parts


pitch = load_parts(os.path.join(ROOT, "stadium-unpacked.glb"), lambda n: n.startswith("000-full-marked"))
figures = load_parts(os.path.join(ROOT, "mannequin-unpacked.glb"))
stands = load_parts(os.path.join(ROOT, "stadium-unpacked.glb"), lambda n: n.startswith(("009-seated", "010-stand")))
goal = load_parts(os.path.join(ROOT, "stadium-unpacked.glb"), lambda n: n.startswith("001-full-size-goal"))


def add_part(viewer, name, xyz, indices, color):
    device = viewer.device
    normals = trimesh.Trimesh(vertices=xyz, faces=indices.reshape(-1, 3), process=False).vertex_normals
    viewer.log_mesh(name, wp.array(xyz.astype(np.float32), dtype=wp.vec3, device=device),
                    wp.array(indices, dtype=wp.int32, device=device),
                    normals=wp.array(normals.astype(np.float32), dtype=wp.vec3, device=device),
                    hidden=True, backface_culling=False)
    viewer.log_instances(name + "_instances", name,
                         wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                         wp.array([(1., 1., 1.)], dtype=wp.vec3, device=device),
                         wp.array([tuple(color)], dtype=wp.vec3, device=device),
                         wp.array([(0.5, 0., 0., 0.)], dtype=wp.vec4, device=device))


original_build = recorder.NewtonVideoRecorder._build_viewer


def build_viewer(self):
    original_build(self)
    v = self._viewer
    for i, (_, xyz, indices, color) in enumerate(pitch):
        xyz = xyz.copy()
        xyz[:, :2] *= 0.25
        xyz[:, 2] -= 0.045
        add_part(v, f"pitch{i}", xyz, indices, color)
    for j, (x, y) in enumerate([(2., 2.), (4., 3.3), (-1., 4.5), (5., -0.5)]):
        for i, (_, xyz, indices, color) in enumerate(figures):
            add_part(v, f"figure{j}_{i}", xyz + (x, y, 0.02), indices, color)
    for j, y in enumerate((-4., 0., 4.)):
        for i, (_, xyz, indices, color) in enumerate(stands):
            add_part(v, f"stand{j}_{i}", xyz + (48., y - 10., 0.), indices, color)
    for i, (_, xyz, indices, color) in enumerate(goal):
        add_part(v, f"goal{i}", xyz + (0., -44., 0.), indices, color)


recorder.NewtonVideoRecorder._build_viewer = build_viewer
sys.argv[0] = os.path.join(MIMICKIT, "mimickit/run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
