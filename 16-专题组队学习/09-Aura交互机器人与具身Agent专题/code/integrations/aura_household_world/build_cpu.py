"""Author USD and validate composition on CPU. No task/model/Kit startup."""
import argparse,json,os
from pathlib import Path
import subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    install=Path(os.environ['ISAACSIM_ROOT'])
    usd=next((install/'extscache').glob('omni.usd.libs-*'))
    if os.environ.get('AURA_WORLD_CPU_USD')!='1':
        env=dict(os.environ,AURA_WORLD_CPU_USD='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        env['LD_LIBRARY_PATH']=':'.join([str(Path(sys.base_prefix)/'lib'),str(Path(sys.prefix)/'lib'),str(usd/'bin'),str(install/'kit'),env.get('LD_LIBRARY_PATH','')])
        return subprocess.call([sys.executable,'-B',__file__,*sys.argv[1:]],env=env)
    sys.path.insert(0,str(usd));sys.path.insert(0,str(Path(__file__).parent))
    from scene_builder import author
    config=json.loads(a.config.read_text())
    if config['execution_enabled'] is not False:raise ValueError('CPU preparation requires execution disabled')
    result=author(config,a.output)
    from pxr import Usd,UsdGeom,UsdUtils
    stage=Usd.Stage.Open(str(a.output/'household-interface.usda'))
    errors=[]
    if not stage.GetPrimAtPath('/HouseholdInterface/Aura/Model'):errors.append('Aura model missing')
    if not result['screen_meshes']:errors.append('Aura display missing')
    if not result['human_skeletons']:errors.append('Human skeleton missing')
    if len(result['cameras'])!=2:errors.append('Observer or Aura camera missing')
    meshes=sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse())
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(a.output/'household-interface.usda'))
    unresolved=[str(x) for x in unresolved]
    result.update(mesh_count=meshes,composed_layers=len(layers),unresolved_dependencies=unresolved,
                  structural_errors=errors,cpu_structure_passed=not errors and not unresolved,
                  execution_enabled=False,full_house_rendered=False)
    with (a.output/'cpu-validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    return 0 if result['cpu_structure_passed'] else 1

if __name__=='__main__':sys.exit(main())
