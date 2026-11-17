"""Read asset structure using USD libraries only. Never create SimulationApp."""
import argparse,json,os
from pathlib import Path
import subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    install=Path(os.environ['ISAACSIM_ROOT'])
    usd=next((install/'extscache').glob('omni.usd.libs-*'))
    if os.environ.get('AURA_WORLD_CPU_USD')!='1':
        env=dict(os.environ,AURA_WORLD_CPU_USD='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        env['LD_LIBRARY_PATH']=':'.join([str(Path(sys.base_prefix)/'lib'),str(Path(sys.prefix)/'lib'),str(usd/'bin'),str(install/'kit'),env.get('LD_LIBRARY_PATH','')])
        return subprocess.call([sys.executable,'-B',__file__,*sys.argv[1:]],env=env)
    sys.path.insert(0,str(usd))
    from pxr import Usd,UsdGeom
    home=Path(os.environ['AURA_HOME_USD'])
    assets=Path(os.environ['ISAAC_PEOPLE_DIR'])
    stage=Usd.Stage.Open(str(home));cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=True)
    found=[]
    for prim in stage.Traverse():
        if 'Aura' in prim.GetName() and (prim.GetParent()==stage.GetDefaultPrim() or prim.GetName()=='Aura_HOME_offset'):
            box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            found.append({'path':str(prim.GetPath()),'bounds':[list(box.GetMin()),list(box.GetMax())]})
    people={}
    for role,relative in [('visitor','Characters/F_Business_02/F_Business_02.usd'),('wave','Animations/stand_idle_wave_loop.skelanim.usd'),('idle','Animations/stand_idle_loop.skelanim.usd')]:
        path=assets/relative
        row={'path':str(path),'exists':path.is_file()}
        if path.is_file():
            s=Usd.Stage.Open(str(path));row.update(default_prim=str(s.GetDefaultPrim().GetPath()) if s.GetDefaultPrim() else None,
                up_axis=str(UsdGeom.GetStageUpAxis(s)),meters_per_unit=UsdGeom.GetStageMetersPerUnit(s),
                skeleton_paths=[str(x.GetPath()) for x in s.Traverse() if x.GetTypeName() in ('Skeleton','SkelRoot','SkelAnimation')][:12])
        people[role]=row
    output={'home_source':str(home),'home_default':str(stage.GetDefaultPrim().GetPath()),'aura_candidates':found,'people':people,'kit_started':False,'gpu_used':False}
    with a.output.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
