"""Build MuJoCo from the CAD mesh manifest, with separate physical proxies."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from model_parameters import P

ROOT = Path(__file__).resolve().parent


def numbers(values):
    return ' '.join(f'{float(x):.9g}' for x in values)


def quaternion(rotation):
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return [xyzw[3], *xyzw[:3]]


def screen_mesh(path):
    count, radius = 96, P.screen_radius*.001
    vertices, faces, texcoords = [], [], []
    for z in (0., -.0003):
        vertices.append([0,0,z]); texcoords.append([.5,.5])
        for i in range(count):
            a = i*2*math.pi/count
            vertices.append([radius*math.cos(a), radius*math.sin(a),z])
            texcoords.append([.5+.5*math.cos(a), .5+.5*math.sin(a)])
    back = count+1
    for i in range(count):
        a,b = 1+i, 1+(i+1)%count
        faces.extend([[0,a,b],[back,back+b,back+a],
                      [a,back+a,back+b],[a,back+b,b]])
    lines = ['v '+numbers(v) for v in vertices]+['vt '+numbers(v) for v in texcoords]
    lines += ['f '+' '.join(f'{i+1}/{i+1}' for i in f) for f in faces]
    path.write_text('\n'.join(lines), encoding='utf-8')


def build():
    manifest = json.loads((ROOT / 'model_manifest.json').read_text(encoding='utf-8'))
    screen_mesh(ROOT / 'assets/meshes/screen_uv.obj')
    model = ET.Element('mujoco', model='Aura_photo_based_twin')
    ET.SubElement(model,'compiler',angle='radian', autolimits='true', inertiafromgeom='false')
    ET.SubElement(model,'option',timestep='0.002',gravity='0 0 -9.81',integrator='implicitfast')
    visual=ET.SubElement(model,'visual')
    ET.SubElement(visual,'global',offwidth='1280',offheight='960')
    ET.SubElement(visual,'quality',shadowsize='2048')
    defaults=ET.SubElement(model,'default')
    ET.SubElement(defaults,'geom',solref='.008 1',solimp='.95 .99 .001',friction='.9 .005 .0001')
    asset=ET.SubElement(model,'asset')
    for name,rgba in {'white':[.9,.91,.9,1],'floor':[.18,.22,.27,1]}.items():
        ET.SubElement(asset,'material',name=name,rgba=numbers(rgba))
    ET.SubElement(asset,'texture',name='screen_texture',type='2d',file='assets/screen/screen.png')
    ET.SubElement(asset,'material',name='screen_material',texture='screen_texture',texuniform='false',
                  emission='1',specular='0',reflectance='0',texrepeat='1 1')
    ET.SubElement(asset,'mesh',name='screen_uv',file='assets/meshes/screen_uv.obj')
    world=ET.SubElement(model,'worldbody')
    ET.SubElement(world,'light',pos='.4 -.5 1.4',dir='-.3 .3 -1',diffuse='.8 .8 .8',castshadow='true')
    ET.SubElement(world,'light',pos='-.5 .7 .8',dir='.3 -.3 -1',diffuse='.5 .55 .65')
    ET.SubElement(world,'geom',name='ground',type='plane',size='3 3 .05',material='floor')
    # An exhibit card, not a live camera fixture.
    ET.SubElement(world,'geom',name='exhibit_block',type='box',pos='.65 .35 .06',size='.06 .06 .06',rgba='.12 .5 .6 1')
    def camera(name,pos,target):
        direction=np.array(pos)-np.array(target)
        z=direction/np.linalg.norm(direction)
        x=np.cross([0,0,1],z); x/=np.linalg.norm(x)
        y=np.cross(z,x)
        ET.SubElement(world,'camera',name=name,pos=numbers(pos),xyaxes=numbers([*x,*y]),fovy='34')
    camera('hero',[.53,-.55,.38],[0,0,.105])
    camera('front',[.63,0,.37],[.025,0,.12])
    camera('side',[.05,-.62,.28],[0,0,.105])
    root=ET.SubElement(world,'body',name='aura',pos='0 0 .003')
    ET.SubElement(root,'freejoint',name='root')
    ET.SubElement(root,'inertial',pos='-.012 0 .082',mass=str(P.chassis_mass_kg),diaginertia='.005 .006 .004')
    ET.SubElement(root,'geom',name='chassis_collision',type='ellipsoid',pos='-.015 0 .112',
                  size='.073 .066 .078',rgba='0 0 0 0',group='3')
    bodies={'chassis':root}
    for side,sign in [('left',1),('right',-1)]:
        body=ET.SubElement(root,'body',name='wheel_'+side,
            pos=numbers([P.wheel_x*.001, sign*P.wheel_y*.001,P.wheel_radius*.001]))
        ET.SubElement(body,'joint',name='wheel_'+side,type='hinge',axis='0 1 0',damping='.015',armature='.00005')
        ET.SubElement(body,'inertial',pos='0 0 0',mass=str(P.wheel_mass_kg),diaginertia='.00016 .00031 .00016')
        ET.SubElement(body,'geom',name='wheel_'+side+'_collision',type='cylinder',
            size=numbers([P.wheel_radius*.001,P.wheel_width*.0005]),quat='.707106781 .707106781 0 0',
            rgba='0 0 0 0',group='3',friction='1.0 .005 .0001',condim='4')
        bodies['wheel_'+side]=body
    caster=ET.SubElement(root,'body',name='caster',pos=numbers([P.caster_x*.001,0,P.caster_radius*.001]))
    ET.SubElement(caster,'joint',name='caster_ball',type='ball',damping='.00001')
    ET.SubElement(caster,'inertial',pos='0 0 0',mass='.025',diaginertia='.00000144 .00000144 .00000144')
    ET.SubElement(caster,'geom',name='caster_collision',type='sphere',size=str(P.caster_radius*.001),
        rgba='0 0 0 0',group='3',friction='.7 .0001 .00001')
    bodies['caster']=caster
    anchors={'chassis':np.zeros(3),'wheel_left':np.array([P.wheel_x,P.wheel_y,P.wheel_radius]),
             'wheel_right':np.array([P.wheel_x,-P.wheel_y,P.wheel_radius]),'caster':np.array([P.caster_x,0,P.caster_radius])}
    for item in manifest['components']:
        ET.SubElement(asset,'mesh',name=item['name'],file=item['mesh'])
        ET.SubElement(bodies[item['moving']],'geom',name=item['name'],type='mesh',mesh=item['name'],
            pos=numbers((np.array(item['position_mm'])-anchors[item['moving']])*.001),
            quat=numbers(quaternion(item['rotation'])),rgba=numbers(item['rgba']),
            contype='0',conaffinity='0',group='1',mass='0')
    screen=manifest['screen']
    ET.SubElement(root,'geom',name='display_pixels',type='mesh',mesh='screen_uv',
        pos=numbers(np.array(screen['position_mm'])*.001),quat=numbers(quaternion(screen['rotation'])),
        material='screen_material',contype='0',conaffinity='0',group='1',mass='0')
    camera_rot=np.array(P.face_rotation)
    ET.SubElement(root,'camera',name='rgb_left',pos=numbers(np.array(P.face_point(P.camera_spacing/2,P.camera_v,7))*.001),
        xyaxes=numbers([*(-camera_rot[:,0]),*camera_rot[:,1]]),fovy='70')
    ET.SubElement(root,'camera',name='rgb_right',pos=numbers(np.array(P.face_point(-P.camera_spacing/2,P.camera_v,7))*.001),
        xyaxes=numbers([*(-camera_rot[:,0]),*camera_rot[:,1]]),fovy='70')
    sensor=ET.SubElement(model,'sensor')
    ET.SubElement(root,'site',name='imu',pos='0 0 .1',size='.002',rgba='0 0 0 0')
    ET.SubElement(sensor,'gyro',name='gyro',site='imu')
    ET.SubElement(sensor,'accelerometer',name='accelerometer',site='imu')
    act=ET.SubElement(model,'actuator')
    for side in ['left','right']:
        ET.SubElement(act,'velocity',name='drive_'+side,joint='wheel_'+side,kv='.12',
                      ctrlrange='-6 6',forcerange='-.3 .3')
    ET.indent(model)
    ET.ElementTree(model).write(ROOT/'aura.xml',encoding='utf-8',xml_declaration=True)
    print('Built aura.xml from CAD geometry; physical parameters remain unmeasured.')


if __name__=='__main__':
    build()
