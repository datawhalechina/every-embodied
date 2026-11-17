"""Run with Blender --background --python; import the same CAD-derived meshes.

Motion keys are a playback of the measured MuJoCo rollout, not an independent
Blender dynamics solver. Display sequence uses real vendor video frames.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import bpy
from mathutils import Matrix, Vector, Quaternion

ROOT=Path(__file__).resolve().parent


def material(name,rgba,metal=0,roughness=.35):
    mat=bpy.data.materials.new(name)
    mat.diffuse_color=rgba
    mat.use_nodes=True
    node=mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value=tuple(v**2.2 for v in rgba[:3])+(rgba[3],)
    node.inputs['Metallic'].default_value=metal
    node.inputs['Roughness'].default_value=roughness
    return mat


def look_at(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def import_lab():
    """Use the same static MJCF room geometry and card UV, not a second layout."""
    world=ET.parse(ROOT/'lab.xml').getroot().find('worldbody')
    imported_names=[]
    for geom in world.findall('geom'):
        name=geom.get('name','')
        if name=='ground': continue
        position=tuple(map(float,geom.get('pos','0 0 0').split()))
        if geom.get('type')=='box':
            bpy.ops.mesh.primitive_cube_add(size=2,location=position)
            obj=bpy.context.object
            obj.scale=tuple(map(float,geom.get('size').split()))
            rgba=tuple(map(float,geom.get('rgba','.5 .5 .5 1').split()))
            obj.data.materials.append(material('lab_'+name,rgba,roughness=.8))
        elif name=='lab_card':
            before=set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(ROOT/'assets/lab/card_uv.obj'),forward_axis='Y',up_axis='Z')
            objects=list(set(bpy.data.objects)-before)
            if len(objects)!=1: raise RuntimeError('Unexpected card mesh import')
            obj=objects[0]; obj.location=position
            obj.rotation_mode='QUATERNION'
            obj.rotation_quaternion=Quaternion(tuple(map(float,geom.get('quat').split())))
            mat=material('virtual_card_image',(1,1,1,1),roughness=1)
            texture=mat.node_tree.nodes.new('ShaderNodeTexImage')
            texture.image=bpy.data.images.load(str(ROOT/'assets/lab/card.png'))
            texture.image.filepath='//assets/lab/card.png'
            mat.node_tree.links.new(texture.outputs['Color'],mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
            obj.data.materials.clear(); obj.data.materials.append(mat)
        else: continue
        obj.name=name
        imported_names.append(name)
    return imported_names


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trajectory',type=Path,default=ROOT/'outputs/mujoco-v2/trajectory.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/blender-v1')
    parser.add_argument('--preview-animation',action='store_true')
    parser.add_argument('--lab',action='store_true',help='Build a separate aura_lab.blend with the MuJoCo room')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    args.output=args.output.resolve()
    args.trajectory=args.trajectory.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'model_manifest.json').read_text(encoding='utf-8'))
    trajectory=json.loads(args.trajectory.read_text(encoding='utf-8'))
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    scene.unit_settings.scale_length=1
    scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=900
    scene.render.resolution_y=900
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.fps=30
    scene.frame_start=1
    scene.frame_end=round(trajectory['simulated_s']*scene.render.fps)
    scene.world.color=(.18,.18,.18)
    scene.view_settings.view_transform='AgX'
    p=manifest['parameters']['values_mm_unless_suffixed']
    root=bpy.data.objects.new('Aura_ROOT_physics_playback',None)
    scene.collection.objects.link(root)
    roots={'chassis':root}
    anchors={'chassis':Vector((0,0,0))}
    for side,sign in [('left',1),('right',-1)]:
        pivot=bpy.data.objects.new('wheel_'+side+'_joint',None)
        scene.collection.objects.link(pivot)
        pivot.parent=root
        pivot.location=(p['wheel_x']*.001,sign*p['wheel_y']*.001,p['wheel_radius']*.001)
        roots['wheel_'+side]=pivot
        anchors['wheel_'+side]=pivot.location.copy()
    roots['caster']=root
    anchors['caster']=Vector((0,0,0))
    mats={}
    for item in manifest['components']:
        key=item['material']
        if key not in mats:
            mats[key]=material(key,item['rgba'],.72 if key=='silver' else 0,
                .8 if key=='rubber' else .25 if key in ('glass','lens') else .32)
        before=set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(ROOT/item['mesh']),forward_axis='Y',up_axis='Z')
        imported=list(set(bpy.data.objects)-before)
        for obj in imported:
            obj.name=item['name']
            obj.data.materials.clear(); obj.data.materials.append(mats[key])
            obj.parent=roots[item['moving']]
            transform=Matrix(item['rotation']).to_4x4()
            transform.translation=Vector(item['position_mm'])*.001-anchors[item['moving']]
            obj.matrix_local=transform
            for poly in obj.data.polygons:
                poly.use_smooth=True

    # Explicit UV disk: never texture a sphere and stretch the round display.
    screen=manifest['screen']
    count=96; r=screen['radius_mm']*.001
    vertices=[(0,0,0)]+[(r*math.cos(2*math.pi*i/count),r*math.sin(2*math.pi*i/count),0) for i in range(count)]
    faces=[(0,1+i,1+(i+1)%count) for i in range(count)]
    mesh=bpy.data.meshes.new('round_screen_UV')
    mesh.from_pydata(vertices,[],faces); mesh.update()
    uv=mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            v=mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv.data[loop_index].uv=(.5+v.x/(2*r),.5+v.y/(2*r))
    display=bpy.data.objects.new('Aura_display_pixels',mesh)
    scene.collection.objects.link(display)
    display.parent=root
    display.matrix_local=Matrix(screen['rotation']).to_4x4()
    display.location=Vector(screen['position_mm'])*.001
    display_mat=bpy.data.materials.new('display_original_nebula_sequence')
    display_mat.use_nodes=True
    nodes=display_mat.node_tree.nodes; nodes.clear()
    tex=nodes.new('ShaderNodeTexImage')
    first_screen=next(iter(sorted((ROOT/'assets/screen/sequence').glob('*.png'))))
    tex.image=bpy.data.images.load(str(first_screen))
    tex.image.filepath='//assets/screen/sequence/'+first_screen.name
    tex.image.source='SEQUENCE'
    screen_info=json.loads((ROOT/'assets/screen/provenance.json').read_text(encoding='utf-8'))
    tex.image_user.frame_duration=screen_info['frames']
    tex.image_user.frame_start=1
    tex.image_user.use_cyclic=True
    tex.image_user.use_auto_refresh=True
    emission=nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value=.9
    output=nodes.new('ShaderNodeOutputMaterial')
    display_mat.node_tree.links.new(tex.outputs['Color'],emission.inputs['Color'])
    display_mat.node_tree.links.new(emission.outputs[0],output.inputs['Surface'])
    display.data.materials.append(display_mat)

    # Root pose and wheel angles come from the actual MuJoCo trajectory.
    for frame in trajectory['frames']:
        index=1+round(frame['time']*scene.render.fps); q=frame['qpos']
        root.location=q[:3]
        root.rotation_mode='QUATERNION'; root.rotation_quaternion=q[3:7]
        root.keyframe_insert(data_path='location',frame=index)
        root.keyframe_insert(data_path='rotation_quaternion',frame=index)
        for name,angle in [('wheel_left',q[7]),('wheel_right',q[8])]:
            roots[name].rotation_euler=(0,angle,0)
            roots[name].keyframe_insert(data_path='rotation_euler',frame=index)
    room_names=import_lab() if args.lab else []
    bpy.ops.mesh.primitive_plane_add(size=6.4 if args.lab else 200,location=(0,0,-.0005))
    floor=bpy.context.object; floor.name='studio_floor'
    floor.data.materials.append(material('studio_warm_gray',(.38,.4,.42,1),roughness=.8))
    for name,pos,power,size in [('key',(.3,-.4,.75),25,.55),('fill',(.1,.6,.45),15,.5),('rim',(-.5,-.1,.65),30,.4)]:
        lamp=bpy.data.lights.new(name,'AREA'); lamp.energy=power; lamp.shape='DISK'; lamp.size=size
        obj=bpy.data.objects.new(name,lamp); scene.collection.objects.link(obj); obj.location=pos
        look_at(obj,(0,0,.12))
    if args.lab:
        lamp=bpy.data.lights.new('room_softbox','AREA'); lamp.energy=140; lamp.size=2.4
        obj=bpy.data.objects.new('room_softbox',lamp); scene.collection.objects.link(obj)
        obj.location=(.5,0,2.4); look_at(obj,(.5,0,0))
    camera_data=bpy.data.cameras.new('hero_camera')
    camera=bpy.data.objects.new('hero_camera',camera_data)
    scene.collection.objects.link(camera)
    camera.location=(.44,-.51,.32); look_at(camera,(0,0,.108))
    camera_data.lens=55; camera_data.clip_start=.005; camera_data.clip_end=1000
    scene.camera=camera
    scene['fidelity']='photo-based dimensions, not measured 1:1'
    scene['dynamics_source']='MuJoCo contact rollout playback; not Blender rigid-body validation'
    scene['screen_source']='Project-authored course UI states; see screen provenance.json'
    if args.lab: scene['environment_source']='Static world geoms from lab.xml; generated virtual card'
    blend_path=ROOT/('aura_lab.blend' if args.lab else 'aura_twin.blend')
    scene.frame_set(7)
    scene.render.filepath=str(args.output/'blender-hero.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)
    # Front view also checks both camera lenses and the display placement.
    camera.location=(.59,0,.32); look_at(camera,(.015,0,.12))
    scene.render.filepath=str(args.output/'blender-front.png')
    bpy.ops.render.render(write_still=True)
    camera.location=(.44,-.51,.32); look_at(camera,(0,0,.108))
    if args.lab:
        scene.render.resolution_x=1200; scene.render.resolution_y=800
        camera.location=(.34,-1.65,.94); look_at(camera,(.35,0,.30)); camera_data.lens=43
        scene.render.filepath=str(args.output/'blender-room.png')
        bpy.ops.render.render(write_still=True)
    if args.preview_animation:
        sequence=args.output/'frames'; sequence.mkdir()
        scene.render.resolution_percentage=66
        for number,index in enumerate(range(1,scene.frame_end+1,5),1):
            scene.frame_set(index)
            scene.render.filepath=str(sequence/f'{number:04d}.png')
            bpy.ops.render.render(write_still=True)
    scene.frame_set(7)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report={'blender_version':bpy.app.version_string,'mesh_components':len(manifest['components']),
        'timeline_frames':scene.frame_end,'screen_sequence_frames':screen_info['frames'],
        'screen_fps':screen_info['fps'],'timeline_fps':scene.render.fps,
        'units':'meters','dynamics':'MuJoCo trajectory playback',
        'physical_robot_access':False,'precision':'provisional_dimensions',
        'lab_world_geoms':room_names,'blend_file':blend_path.name}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
