"""Reuse fingerprinted, physically replayed cup inputs without reusing a policy.

The target source is copied once and frozen. Existing targets are verified without
changing stage state, which allows this command to precede a resumed notebook.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--target-run', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    old, root, code, assets = (p.resolve() for p in (args.source_run, args.target_run, args.source_code, args.assets))
    assert old != root and (code / 'course_run.py').is_file() and assets.is_dir()
    old_state = json.loads((old / 'pipeline_state.json').read_text())
    assert old_state['status'] == 'complete'
    assert old_state['completed']['collect']['saved_episodes'] == 40
    assert old_state['completed']['audit']['physical_success_count'] == 40
    existing = (root / 'pipeline_state.json').exists()
    if existing:
        migration = json.loads((root / 'input-migration.json').read_text())
        assert Path(migration['source_run']).resolve() == old
        assert Path(migration['destination_run']).resolve() == root
    else:
        # Refuse partially populated targets rather than overwrite failed runs.
        assert not root.exists() or not any(root.iterdir()), 'Target must be empty or have verified migration evidence'
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(code, root / 'source', ignore=shutil.ignore_patterns('asset', '__pycache__', '*.pyc'))
        (root / 'source/asset').symlink_to(assets, target_is_directory=True)
        for name in ('dataset', 'collection-videos'):
            assert (old / name).is_dir()
            (root / name).symlink_to(old / name, target_is_directory=True)
        for name in ('replay.jsonl', 'replay-summary.json'):
            shutil.copy2(old / name, root / name)
    for name in ('dataset', 'collection-videos'):
        assert (root / name).resolve() == (old / name).resolve()
    assert (root / 'source/asset').resolve() == assets
    # Compare all executable source bytes with the selected recipe before import.
    def sources(directory):
        return {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*.py')
                if '__pycache__' not in p.parts and 'asset' not in p.relative_to(directory).parts}
    fingerprints = sources(root / 'source')
    assert fingerprints == sources(code), 'Selected source differs from frozen target'
    os.environ['CUP_RUN_ROOT'] = str(root)
    spec = importlib.util.spec_from_file_location('verified_cup_recipe', root / 'source/course_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RECIPE['steps'] == 20000 and module.RECIPE['use_vae'] is False
    module.validate('collect')
    module.validate('audit')
    contract = {'recipe': module.RECIPE, 'sources': fingerprints}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    files = [{'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted((old / 'dataset/data').rglob('*.parquet'))]
    assert len(files) == 40
    if existing:
        assert migration['source_contract_hash'] == old_state['contract_hash']
        assert migration['new_contract_hash'] == contract_hash
        assert migration['input_parquet'] == files, 'Verified input data changed'
        current = json.loads((root / 'pipeline_state.json').read_text())
        assert current['contract_hash'] == contract_hash
        assert json.loads((root / 'recipe.json').read_text()) == contract
        for name in ('replay.jsonl', 'replay-summary.json'):
            assert sha(root / name) == sha(old / name)
        print('Verified existing migration; stage state and trained outputs retained.')
        return
    migration = {'source_run': str(old), 'source_contract_hash': old_state['contract_hash'],
                 'destination_run': str(root), 'new_contract_hash': contract_hash,
                 'scope': 'Reuse40 verified input episodes and replay evidence; no training/evaluation results reused',
                 'input_parquet': files}
    write(root / 'input-migration.json', migration)
    write(root / 'recipe.json', contract)
    write(root / 'pipeline_state.json', {'completed': {k: old_state['completed'][k] for k in ('collect', 'audit')},
          'contract_hash': contract_hash, 'status': 'inputs_migrated', 'stage': 'train',
          'input_migration': str(root / 'input-migration.json')})
    print('Migrated 40 verified input episodes; fresh training can start.')


if __name__ == '__main__':
    main()
