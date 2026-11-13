"""Execute a course notebook with an explicit profile and preserve failed output."""
from pathlib import Path
import argparse
import os
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profile', type=Path)
parser.add_argument('--kernel', default='python3')
parser.add_argument('--enable-workloads', action='store_true')
parser.add_argument('--timeout', type=int, default=-1,
                    help='Seconds per cell; a negative value permits long training cells.')
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Choose a fresh output path to preserve previous execution evidence.')
notebook = nbformat.read(args.notebook, as_version=4)
workflow = notebook.metadata.get('amd_course', {}).get('workflow')
if args.profile:
    if not workflow:
        raise ValueError('A workflow metadata value is required when selecting a profile.')
    os.environ['AMD_' + workflow.upper() + '_CONFIG'] = str(args.profile.resolve())
if args.enable_workloads:
    os.environ['AMD_RUN_FULL'] = '1'
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for stage in ['PREFLIGHT', 'PREPARE', 'TRAIN', 'BASELINE', 'EVAL']:
                cell.source = cell.source.replace('RUN_' + stage + ' = False', 'RUN_' + stage + ' = True')
client = NotebookClient(notebook, timeout=None if args.timeout < 0 else args.timeout, kernel_name=args.kernel,
                        resources={'metadata': {'path': str(args.notebook.resolve().parent)}})
args.output.parent.mkdir(parents=True, exist_ok=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, args.output)
print(args.output.resolve())
