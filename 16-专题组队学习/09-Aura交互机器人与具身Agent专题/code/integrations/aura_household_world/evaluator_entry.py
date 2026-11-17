"""Reuse native counter/private storage in this world's isolated cache only."""
import json,os,sys
from pathlib import Path

def main():
    from .agent import require_execution
    config=json.loads(Path(os.environ['AURA_HOUSEHOLD_WORLD_CONFIG']).read_text())
    require_execution(config,explicit_run=os.environ.get('AURA_HOUSEHOLD_EXPLICIT_RUN')=='1')
    reference=Path(__file__).resolve().parents[1]/'aura_roboharness/runtime_20261113/reference'
    sys.path.insert(0,str(reference))
    import private_storage,native_counter_entry_compact_v2
    root=Path(config['remote_root']).resolve()
    cache=Path(os.environ['OMNIGIBSON_APPDATA_PATH']).resolve()
    if not cache.is_relative_to(root):raise ValueError('Evaluator cache must belong to this world')
    private_storage.ROOT=root
    native_counter_entry_compact_v2.main()

if __name__=='__main__':main()
