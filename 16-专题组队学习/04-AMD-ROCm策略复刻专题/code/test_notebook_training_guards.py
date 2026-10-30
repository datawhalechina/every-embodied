"""Small fixtures for reuse guards and repeatable notebook instrumentation."""
import ast, hashlib, importlib.util, json, os, pathlib, sys, tempfile, types
from unittest.mock import patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
results=[]
for name in ['cup_pipeline','cup_pipeline_deterministic']:
 spec=importlib.util.spec_from_file_location(name, ROOT/'code'/name/'course_run.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 with tempfile.TemporaryDirectory() as temp:
  root=pathlib.Path(temp);mod.RUN=root/'run';mod.SOURCE=root/'source';(mod.SOURCE/'asset').mkdir(parents=True)
  mod.validate=lambda stage:{'fixture':True}
  ns={'__name__':'fixture_inline','counter':0}
  exec(compile('def helper():\n    return 1\ndef callback(argv):\n    global counter\n    counter += helper()\n','<ipython-input-1>', 'exec'),ns)
  h=mod.callback_fingerprint(ns['callback'])
  with patch.dict(os.environ, {'LEROBOT_SRC':str(root)}), patch.object(mod.shutil,'disk_usage',return_value=types.SimpleNamespace(free=100*2**30)):
   mod.run('collect',stage_callable=ns['callback'],implementation_sha256=h)
   mod.run('collect',stage_callable=ns['callback'],implementation_sha256=h)
   assert ns['counter']==1
   # A new cell execution counter must not invalidate identical executable code.
   exec(compile('def helper():\n    return 1\n','<ipython-input-99>', 'exec'),ns)
   assert mod.callback_fingerprint(ns['callback'])==h
   exec('def helper():\n    return 2\n',ns)
   assert mod.callback_fingerprint(ns['callback'])!=h
   try:mod.run('collect',stage_callable=ns['callback'])
   except RuntimeError as exc:assert 'Inline implementation changed' in str(exc)
   else:raise AssertionError('Changed inline helper reused an old result')
  results.append({'test':name+' inline implementation change and reuse','passed':True})

# Compile exactly the live notebook adapter functions, then replace only expensive IO.
p=ROOT/'03-DexJoCo灵巧手/03-Notebook/18_dexjoco_amd_train_eval.ipynb'
nb=json.loads(p.read_text(encoding='utf-8'))
cell=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'def _train_with_patches(c, cfg, script):' in ''.join(c['source']))
tree=ast.parse(cell); funcs=[node for node in tree.body[0].body if isinstance(node,ast.FunctionDef) and node.name in {'train','_train_with_patches'}]
ns={'Path':pathlib.Path,'os':os,'sys':sys,'json':json,'time':__import__('time')}
exec(compile(ast.Module(body=funcs,type_ignores=[]),'<notebook-adapter>', 'exec'),ns)
class StopFixture(Exception):pass
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);ck=root/'checkpoints';(ck/'1').mkdir(parents=True)
 (root/'prepared.json').write_text(json.dumps({'normalization_frames':None,'stats_path':'fixture','normalization_sha256':'ok'}))
 ns.update(require_files=lambda *a:None,sha256=lambda p:'ok',disk_gate=lambda *a:None,memory_gate=lambda *a:None,reconcile_metrics=lambda *a:{},write_json=lambda *a:None,source_root=lambda c:root)
 cfg=types.SimpleNamespace(checkpoint_dir=ck,trainable_filter=None)
 ns['get_config']=lambda c:cfg
 class Params:
  def filter(self,f):return self
 state=types.SimpleNamespace(step=2,params=Params())
 loader=types.SimpleNamespace(create_data_loader=lambda *a,**k:list(range(10)))
 checkpoints=types.SimpleNamespace(restore_state=lambda *a,**k:state, save_state=lambda *a,**k:None)
 wandb=types.SimpleNamespace(log=lambda *a,**k:None)
 sharding=types.SimpleNamespace(make_mesh=lambda n:None)
 nnx_utils=types.SimpleNamespace(state_map=lambda *a:None)
 def wb_init(*a,**k):wandb.log=lambda *a,**k:None
 space={'nnx_utils':nnx_utils,'sharding':sharding,'_data_loader':loader,'_checkpoints':checkpoints,'wandb':wandb,'init_train_state':lambda *a,**k:(state,None),'init_wandb':wb_init}
 class Proxy:
  def __getattr__(self,n):return space[n]
  def __setattr__(self,n,v):space[n]=v
 proxy=Proxy();ns['load_script']=lambda *a:proxy
 calls=[]
 def main(config):
  expected=2 if current['resume'] else 0
  assert next(iter(proxy._data_loader.create_data_loader()))==expected
  if current['resume']:assert proxy._checkpoints.restore_state().step==2
  proxy.init_wandb()
  calls.append(expected)
  raise StopFixture('Stop after exercising instrumentation; no model was created')
 space['main']=main
 fakejax=types.SimpleNamespace(make_mesh=lambda n:None,tree=types.SimpleNamespace(leaves=lambda x:[np.array([1.0])]),device_get=lambda x:x)
 targets=[(proxy,'init_train_state'),(proxy,'init_wandb'),(loader,'create_data_loader'),(checkpoints,'restore_state'),(checkpoints,'save_state'),(wandb,'log'),(sharding,'make_mesh'),(nnx_utils,'state_map')]
 originals=[getattr(o,k) for o,k in targets]
 with patch.dict(sys.modules,{'jax':fakejax}):
  for resume in [False,True,True]:
   current={'run_root':str(root),'resume':resume,'steps':8,'normalization_frames':None}
   before=(pathlib.Path.cwd(),dict(os.environ),sys.path[:])
   try:ns['train'](current)
   except StopFixture:pass
   else:raise AssertionError('Fixture must stop before training')
   assert all(getattr(o,k) is v for (o,k),v in zip(targets,originals))
   assert before==(pathlib.Path.cwd(),dict(os.environ),sys.path[:])
 assert calls==[0,2,2]
 # Exercise the successful-return finally path independently of any model.
 original_body=ns['_train_with_patches']
 def success_body(c,cfg,script):
  script.init_wandb=lambda *a:None
  script._data_loader.create_data_loader=lambda *a:[]
  return {'fixture_success':True}
 ns['_train_with_patches']=success_body
 assert ns['train'](current)=={'fixture_success':True}
 assert all(getattr(o,k) is v for (o,k),v in zip(targets,originals))
 ns['_train_with_patches']=original_body
 results.append({'test':'live notebook namespace repeat, resume twice, exception and successful-return cleanup','passed':True,'observed_skip_batches':calls})

# Validate non-finite loss is stopped before backward/update with a scalar fixture.
import torch
for loss in [torch.tensor(float('nan')),torch.tensor(float('inf'))]:
 try:
  if not torch.isfinite(loss).all():raise FloatingPointError('fixture')
 except FloatingPointError:pass
 else:raise AssertionError('Non-finite loss accepted')
p=torch.nn.Parameter(torch.tensor(1.0));p.grad=torch.tensor(float('nan'))
try:torch.nn.utils.clip_grad_norm_([p],1.0,error_if_nonfinite=True)
except RuntimeError:pass
else:raise AssertionError('Non-finite gradient accepted')
results.append({'test':'non-finite scalar loss and gradient guards','passed':True})
print(json.dumps(results,indent=2))
