"""Future evaluator wrapper: same scene/physics clock for visitor, Aura and R1.

Imported only by an explicitly started evaluator. CPU preparation never imports
this module. The current world config keeps execution disabled.
"""
import json,math,os
from pathlib import Path
from behavior_interface_eval_test.official_rgbd_wrapper import OfficialRGBDFullResWrapper


class HouseholdRGBDWrapper(OfficialRGBDFullResWrapper):
    def __init__(self,env):
        from .agent import require_execution
        self.config=json.loads(Path(os.environ['AURA_HOUSEHOLD_WORLD_CONFIG']).read_text())
        require_execution(self.config,explicit_run=os.environ.get('AURA_HOUSEHOLD_EXPLICIT_RUN')=='1')
        super().__init__(env)
        from .scene_builder import compose
        import omnigibson as og
        from pxr import UsdGeom,Gf
        self.root=str(env.scene.prim_path)+'/AuraHousehold'
        if og.sim.stage.GetPrimAtPath(self.root):raise ValueError('Scene already contains the interface layer')
        with og.sim.editing_usd():
            self.scene_report=compose(og.sim.stage,self.config,self.root)
            self.layer=UsdGeom.Xformable(og.sim.stage.GetPrimAtPath(self.root))
            self.translation=self.layer.AddTranslateOp();self.rotation=self.layer.AddRotateZOp()
        self.counter=0;self._phase=None
        self._place()
        self.capture_root=Path(os.environ['AURA_HOUSEHOLD_CAPTURE_ROOT'])
        self.capture_root.mkdir(parents=True,exist_ok=False)
        self.annotators={}
        import omni.replicator.core as rep
        with og.sim.editing_usd():
            for name,path in self.scene_report['cameras'].items():
                product=rep.create.render_product(path,(960,720) if name=='observer' else (640,480))
                rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
                self.annotators[name]=(rgb,product)
        og.sim.update_handles()

    def _place(self):
        import omnigibson as og
        from pxr import Gf
        robot=self._route_robot(self.env);pos,quat=robot.get_position_orientation()
        x,y,z,w=[float(v) for v in quat]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        with og.sim.editing_usd():
            self.translation.Set(Gf.Vec3d(float(pos[0]),float(pos[1]),0.))
            self.rotation.Set(math.degrees(yaw))

    def _feedback(self,raw):
        import omnigibson as og
        from pxr import UsdShade,Sdf
        statefile=Path(os.environ['AURA_HOUSEHOLD_STATE_FILE'])
        if statefile.exists():
            try:phase=json.loads(statefile.read_text())['phase']
            except (OSError,ValueError,KeyError):phase='incomplete'
            if phase not in {'ready','prepared_not_started','working','verifying','completed','incomplete'}:phase='incomplete'
            if phase!=self._phase:
                texture=UsdShade.Shader(og.sim.stage.GetPrimAtPath(self.root+'/Looks/AuraStatus/Texture'))
                if texture:
                    with og.sim.editing_usd():
                        texture.GetInput('file').Set(Sdf.AssetPath(str(Path(self.config['screen_texture_dir'])/(phase+'.png'))))
                self._phase=phase
        if self.counter%3:return
        from PIL import Image
        from .recording import record_frame
        views={name:ann.get_data() for name,(ann,_) in self.annotators.items()}
        def find_head(value,prefix=''):
            if not isinstance(value,dict):return
            for key,item in value.items():
                full=prefix+'/'+str(key)
                if (str(key).endswith('::rgb') or str(key)=='rgb') and 'zed_link' in full:views['r1_head']=item
                elif isinstance(item,dict):find_head(item,full)
        find_head(raw)
        record_frame(self.capture_root,self.counter,views)

    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs);self.counter=0;self._place()
        import omnigibson as og
        og.sim.render()  # Render tick0 without advancing physics.
        self._feedback(result[0] if isinstance(result,tuple) else result)
        return result

    def step(self,action,*args,**kwargs):
        result=super().step(action,*args,**kwargs)
        self.counter+=1
        self._feedback(result[0])
        return result
