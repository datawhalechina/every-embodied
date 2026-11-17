"""CPU USD authoring: Aura + human + cameras, composed into one task scene.

The household and R1 are loaded once by BEHAVIOR. This file adds an asset layer
to that same scene; it does not introduce a second simulator or task robot.
"""
import json,math
from pathlib import Path


def compose(stage,config,root_path='/HouseholdInterface'):
    from pxr import Usd,UsdGeom,UsdShade,UsdSkel,UsdPhysics,Sdf,Gf
    root=UsdGeom.Xform.Define(stage,root_path);root.GetPrim().SetCustomDataByKey('sourceKind','prepared_scene_assets_not_rollout')
    aura=UsdGeom.Xform.Define(stage,root_path+'/Aura')
    aura.AddTranslateOp().Set(Gf.Vec3d(*config['placement']['aura_offset_m']))
    body=UsdGeom.Xform.Define(stage,root_path+'/Aura/Model')
    body.GetPrim().GetReferences().AddReference(config['assets']['aura_home'],config['assets']['aura_prim'])
    body.ClearXformOpOrder()  # Remove only the referenced old home's placement.
    body.AddTranslateOp(opSuffix='localOrigin').Set(Gf.Vec3d(0,0,0))
    capsule=UsdGeom.Cylinder.Define(stage,root_path+'/Aura/Collision')
    capsule.CreateRadiusAttr(config['placement']['aura_radius_m']);capsule.CreateHeightAttr(.20)
    capsule.AddTranslateOp().Set(Gf.Vec3d(0,0,.10));UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    capsule.CreatePurposeAttr('guide')
    human=UsdGeom.Xform.Define(stage,root_path+'/Visitor')
    human.AddTranslateOp().Set(Gf.Vec3d(*config['placement']['visitor_offset_m']))
    human.AddRotateZOp().Set(90.)
    visual=stage.DefinePrim(root_path+'/Visitor/Model','Xform')
    visual.GetReferences().AddReference(config['assets']['visitor'])
    mdl=Path(config['assets'].get('omnipbr_mdl',''))
    material_overrides=[]
    if mdl.is_file():
        for prim in Usd.PrimRange(visual):
            for attr in prim.GetAttributes():
                value=attr.Get()
                if isinstance(value,Sdf.AssetPath) and value.path=='OmniPBR.mdl':
                    attr.Set(Sdf.AssetPath(str(mdl)));material_overrides.append(str(attr.GetPath()))
    animation=stage.DefinePrim(root_path+'/Visitor/Idle','SkelAnimation')
    animation.GetReferences().AddReference(config['assets']['idle_animation'])
    skeletons=[]
    for prim in Usd.PrimRange(visual):
        if prim.IsA(UsdSkel.Skeleton):
            UsdSkel.BindingAPI.Apply(prim).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
            skeletons.append(str(prim.GetPath()))
    human_collision=UsdGeom.Capsule.Define(stage,root_path+'/Visitor/Collision')
    human_collision.CreateRadiusAttr(config['placement']['human_radius_m'])
    human_collision.CreateHeightAttr(config['placement']['human_height_m']-2*config['placement']['human_radius_m'])
    human_collision.AddTranslateOp().Set(Gf.Vec3d(0,0,config['placement']['human_height_m']/2))
    human_collision.CreatePurposeAttr('guide');UsdPhysics.CollisionAPI.Apply(human_collision.GetPrim())
    # Screen material belongs to the new layer, never changes the old asset file.
    mat=UsdShade.Material.Define(stage,root_path+'/Looks/AuraStatus')
    shader=UsdShade.Shader.Define(stage,root_path+'/Looks/AuraStatus/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.05,.4,.65))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.01))
    if config.get('screen_texture_dir'):
        uv=UsdShade.Shader.Define(stage,root_path+'/Looks/AuraStatus/UV');uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture=UsdShade.Shader.Define(stage,root_path+'/Looks/AuraStatus/Texture');texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(config['screen_texture_dir'])/'ready.png')))
        texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        shader.GetInput('emissiveColor').ConnectToSource(texture.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    screens=[]
    for prim in Usd.PrimRange(body.GetPrim()):
        if prim.IsA(UsdGeom.Mesh) and 'display' in str(prim.GetPath()).lower():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            screens.append(str(prim.GetPath()))
    def camera(name,position,target):
        c=UsdGeom.Camera.Define(stage,root_path+'/Cameras/'+name)
        c.CreateFocalLengthAttr(20.);c.CreateClippingRangeAttr(Gf.Vec2f(.03,100.))
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
        c.AddTransformOp().Set(matrix)
        return str(c.GetPath())
    aura_pos=config['placement']['aura_offset_m'];visitor=config['placement']['visitor_offset_m']
    cams={'observer':camera('Observer',config['placement']['observer_offset_m'],[.35,-.5,.7]),
          'aura_head':camera('AuraHead',[aura_pos[0],aura_pos[1]-.08,.15],[visitor[0],visitor[1],1.0])}
    return {'root':root_path,'screen_meshes':screens,'human_skeletons':skeletons,'cameras':cams,'material_path_overrides':material_overrides,
            'same_scene_attachment':True,'physics_layout_validated':False,'human_animation_playback_validated':False}


def author(config,output):
    from pxr import Usd,UsdGeom,Sdf
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    textures=output/'screens';textures.mkdir()
    for phase,title in [('ready','Ready'),('prepared_not_started','Listening'),('working','Working'),('verifying','Checking'),('completed','Done'),('incomplete','Paused')]:
        image=Image.new('RGB',(512,512),(8,17,31));draw=ImageDraw.Draw(image)
        try:font=ImageFont.truetype('DejaVuSans.ttf',42)
        except OSError:font=ImageFont.load_default()
        for x in (176,336):draw.rounded_rectangle((x-24,145,x+24,233),radius=23,fill=(94,224,247))
        draw.arc((190,225,322,315),0,180,fill=(94,224,247),width=8)
        draw.text((256,382),title,anchor='mm',font=font,fill=(224,245,249))
        image.save(textures/(phase+'.png'))
    config=dict(config,screen_texture_dir=str(textures.resolve()),assets=dict(config['assets']))
    # Flatten a new local human layer so CPU dependency resolution does not
    # need Kit's implicit MDL search path. The installed character is read only.
    human_stage=Usd.Stage.CreateInMemory()
    visitor=human_stage.DefinePrim('/Visitor','Xform');visitor.GetReferences().AddReference(config['assets']['visitor'])
    human_stage.SetDefaultPrim(visitor)
    mdl=Path(config['assets']['omnipbr_mdl'])
    if not mdl.is_file():raise FileNotFoundError(mdl)
    for prim in human_stage.Traverse():
        for attr in prim.GetAttributes():
            value=attr.Get()
            if isinstance(value,Sdf.AssetPath) and value.path=='OmniPBR.mdl':attr.Set(Sdf.AssetPath(str(mdl)))
    human_stage.Flatten().Export(str(output/'visitor-prepared.usdc'))
    config['assets']['visitor']=str((output/'visitor-prepared.usdc').resolve())
    stage=Usd.Stage.CreateNew(str(output/'household-interface.usda'))
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(30)
    report=compose(stage,config);stage.SetDefaultPrim(stage.GetPrimAtPath(report['root']))
    stage.GetRootLayer().Save()
    report.update(simulator_started=False,model_calls=0,source='CPU USD composition',
                  host_scene=config['scene'],host_task=config['task_name'],r1_loaded_by='BEHAVIOR evaluator, once')
    with (output/'scene-report.json').open('x') as f:json.dump(report,f,indent=2)
    with (output/'composed-config.json').open('x') as f:json.dump(config,f,indent=2)
    return report
