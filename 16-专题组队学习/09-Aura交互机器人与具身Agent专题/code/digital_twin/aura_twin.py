"""Parametric, STEP-first Aura exterior reconstruction; dimensions are provisional.

Generator contract: gen_step() -> labeled assembly. Simulation meshes are derived
from exactly the same local BREP solids, never independently re-sculpted.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from build123d import (Box, Color, Compound, Cylinder, Keep, Location, Matrix,
                      Plane, Solid, Sphere)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model_parameters import P

PALETTE = {'shell': [.90, .91, .90, 1], 'silver': [.55, .59, .62, 1],
           'panel': [.055, .065, .075, 1], 'rubber': [.73, .70, .61, 1],
           'hub': [.92, .92, .89, 1], 'lens': [.012, .018, .025, 1],
           'glass': [.018, .027, .034, 1], 'blue': [.06, .35, .65, 1]}


def local_location(pos, rotation):
    return Plane(origin=tuple(pos), x_dir=tuple(np.array(rotation)[:, 0]),
                 z_dir=tuple(np.array(rotation)[:, 2])).location


def components():
    records = []
    identity = np.eye(3).tolist()
    def add(name, shape, material, pos=(0, 0, 0), rotation=None, moving='chassis'):
        shape.label = name
        shape.color = Color(*PALETTE[material])
        records.append({'name': name, 'solid': shape, 'material': material,
            'position_mm': list(pos), 'rotation': rotation or identity, 'moving': moving})

    body = Solid.make_sphere(1).transform_geometry(Matrix([
        [P.body_rx, 0, 0, 0], [0, P.body_ry, 0, 0],
        [0, 0, P.body_rz, P.body_z], [0, 0, 0, 1]]))
    normal = np.array([math.cos(P.tilt), 0, math.sin(P.tilt)])
    cut_point = np.array([0, 0, P.body_z]) + normal * P.face_cut
    body = body.split(Plane(origin=tuple(cut_point), z_dir=tuple(normal)), Keep.BOTTOM)
    add('white_shell', body, 'shell')
    face_r = P.face_rotation
    add('face_silver_rim', Cylinder(P.face_radius, 3) - Cylinder(P.face_radius-3.2, 5),
        'silver', P.face_point(depth=1.5), face_r)
    add('dark_faceplate', Cylinder(P.face_radius-3.2, 2.0),
        'panel', P.face_point(depth=1.7), face_r)
    add('screen_rim', Cylinder(P.screen_radius+2, 1.4)-Cylinder(P.screen_radius, 3),
        'lens', P.face_point(v=P.screen_v, depth=3.2), face_r)
    add('screen_glass', Cylinder(P.screen_radius, .7), 'glass',
        P.face_point(v=P.screen_v, depth=3.3), face_r)
    for side, u in [('left', -P.camera_spacing/2), ('right', P.camera_spacing/2)]:
        add('camera_'+side+'_bezel', Cylinder(9.2, 1.6), 'lens',
            P.face_point(u, P.camera_v, 3.4), face_r)
        add('camera_'+side+'_glass', Cylinder(6.7, .6), 'glass',
            P.face_point(u, P.camera_v, 4.4), face_r)
        add('camera_'+side+'_pupil', Cylinder(4.5, .7), 'lens',
            P.face_point(u, P.camera_v, 4.9), face_r)
        add('camera_'+side+'_glint', Cylinder(1.05, .2), 'blue',
            P.face_point(u-1.4, P.camera_v+1.6, 5.3), face_r)

    for side, sign in [('left', 1), ('right', -1)]:
        rot = [[1, 0, 0], [0, 0, -sign], [0, sign, 0]]
        pos = [P.wheel_x, sign*P.wheel_y, P.wheel_radius]
        add('wheel_'+side+'_tire', Cylinder(P.wheel_radius, P.wheel_width)-
            Cylinder(P.wheel_radius-5, P.wheel_width+2), 'rubber', pos, rot, 'wheel_'+side)
        add('wheel_'+side+'_hub', Cylinder(P.wheel_radius-5, P.wheel_width-3),
            'hub', pos, rot, 'wheel_'+side)
        add('wheel_'+side+'_cap', Cylinder(42, .7), 'shell',
            [P.wheel_x, sign*(P.wheel_y+P.wheel_width/2-1), P.wheel_radius], rot, 'wheel_'+side)
    add('caster_tire', Sphere(P.caster_radius), 'rubber',
        [P.caster_x, 0, P.caster_radius], moving='caster')
    add('caster_fork', Box(16, 21, 15), 'shell', [P.caster_x, 0, 27])
    # Small visible speaker perforations are solid dark inserts, not invented internals.
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            if ix*ix+iy*iy > 16:
                continue
            x, y = -22 + 2.4*ix, 2.4*iy
            z = P.body_z + P.body_rz*math.sqrt(1-(x/P.body_rx)**2-(y/P.body_ry)**2)
            add(f'speaker_{ix+4}_{iy+4}', Cylinder(.55, .4), 'panel', [x, y, z])
    add('usb_c_surround', Box(1.5, 10, 4.5), 'silver', [-P.body_rx+1, 0, P.body_z-1])
    add('usb_c_opening', Box(1.7, 7.8, 2.5), 'lens', [-P.body_rx+.2, 0, P.body_z-1])
    return records


def gen_step():
    children = []
    for item in components():
        shape = item['solid'].moved(local_location(item['position_mm'], item['rotation']))
        shape.label = item['name']
        children.append(shape)
    return Compound(label='Aura_photo_based_twin', children=children)


def export_simulation_meshes():
    import trimesh
    folder = ROOT / 'assets/meshes'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for item in components():
        solid = item.pop('solid')
        if not solid.is_valid or solid.volume <= 0:
            raise ValueError('Invalid CAD component: '+item['name'])
        vertices, faces = solid.tessellate(.18, .12)
        xyz = np.array([[v.X, v.Y, v.Z] for v in vertices]) * .001
        mesh = trimesh.Trimesh(vertices=xyz, faces=np.array(faces), process=False)
        filename = item['name'] + '.obj'
        mesh.export(folder / filename)
        item.update(mesh='assets/meshes/'+filename, rgba=PALETTE[item['material']],
                    volume_mm3=solid.volume, cad_valid=True, vertices=len(xyz))
        records.append(item)
    model = gen_step()
    bounds = model.bounding_box()
    report = {'parameters': P.record(), 'mesh_units': 'm', 'cad_units': 'mm',
        'bbox_mm': [bounds.size.X, bounds.size.Y, bounds.size.Z],
        'solids': len(model.solids()), 'components': records,
        'screen': {'position_mm': P.face_point(v=P.screen_v, depth=3.75),
            'rotation': P.face_rotation, 'radius_mm': P.screen_radius},
        'physical_robot_access': False}
    (ROOT / 'model_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('bbox_mm', 'solids', 'physical_robot_access')}))


if __name__ == '__main__':
    export_simulation_meshes()
