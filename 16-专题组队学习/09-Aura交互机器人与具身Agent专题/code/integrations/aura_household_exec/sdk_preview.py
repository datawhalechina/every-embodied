"""Versioned private-workspace correction for missing move_tracked_point previews.

The upstream motion solver and original workspaces remain unchanged. A new
workspace must receive these presentation-only changes before host binding.
"""
import hashlib
from pathlib import Path


def patched_sources(tool_source, media_source):
    before='is_plan_preview = name in PLAN_TOOLS and result.get("ok") and (result.get("marked_image_url") or result.get("gripper_visualization"))'
    after='is_plan_preview = (name in PLAN_TOOLS or (name == "move_tracked_point" and arguments.get("execution_mode", "exec") == "plan")) and result.get("ok") and (result.get("marked_image_url") or result.get("gripper_visualization"))'
    media_before='{"plan_grasp_point_filter_rgbd_lite", "plan_press_point"}'
    media_after='{"plan_grasp_point_filter_rgbd_lite", "plan_press_point", "move_tracked_point"}'
    if tool_source.count(before)!=1 or media_source.count(media_before)!=1:
        raise ValueError('SDK source differs from the audited preview-selection revision')
    tool_source=tool_source.replace(before,after)
    media_source=media_source.replace(media_before,media_after)
    compile(tool_source,'private-tools/util/tool.py','exec')
    compile(media_source,'private-tools/util/media.py','exec')
    return tool_source,media_source


def apply_private_patch(workspace):
    workspace=Path(workspace).resolve()
    if (workspace/'.session.json').exists():raise ValueError('Patch only a fresh, unbound workspace')
    files=[workspace/'tools/util/tool.py',workspace/'tools/util/media.py']
    if any(not p.resolve().is_relative_to(workspace) for p in files):raise ValueError('SDK source leaves the private workspace')
    originals=[p.read_text(encoding='utf-8') for p in files]
    updates=patched_sources(*originals)
    report=[]
    for file,original,updated in zip(files,originals,updates):
        backup=file.with_suffix('.pre-hri-preview.py')
        with backup.open('x',encoding='utf-8') as stream:stream.write(original)
        # A workspace clone may use hardlinks; atomic replacement never edits
        # any source inode shared with another experiment.
        temporary=file.with_suffix('.preview.tmp')
        with temporary.open('x',encoding='utf-8') as stream:stream.write(updated)
        temporary.replace(file)
        report.append({'path':str(file.relative_to(workspace)),
                       'before_sha256':hashlib.sha256(original.encode()).hexdigest(),
                       'after_sha256':hashlib.sha256(updated.encode()).hexdigest()})
    return {'presentation_only':True,'motion_solver_changed':False,'files':report}
