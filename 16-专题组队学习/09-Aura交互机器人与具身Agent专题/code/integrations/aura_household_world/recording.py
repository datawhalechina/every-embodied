"""Same native tick for all views; missing pixels are errors, never substitutes."""
import json
from pathlib import Path


def record_frame(root,tick,views):
    import numpy as np
    from PIL import Image
    required={'observer','r1_head','aura_head'}
    if set(views)!=required:raise ValueError('Three same-world camera views are required')
    arrays={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)) for k,v in views.items()}
    for name,a in arrays.items():
        if a.ndim!=3 or a.shape[-1] not in (3,4) or a.dtype!=np.uint8 or not a.size:
            raise ValueError('Missing actual pixels for '+name)
    root=Path(root);manifest={}
    for name,a in arrays.items():
        folder=root/name;folder.mkdir(exist_ok=True)
        path=folder/f'{tick:07d}.png'
        if path.exists():raise FileExistsError('Never overwrite an existing frame')
        Image.fromarray(a[...,:3]).save(path)
        manifest[name]=str(path.relative_to(root))
    with (root/'frames.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'native_tick':tick,'simulation_time_s':tick/30.,'source':'same_evaluator_step','views':manifest})+'\n')
