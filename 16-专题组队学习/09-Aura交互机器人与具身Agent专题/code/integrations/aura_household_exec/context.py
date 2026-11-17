"""Opt-in, just-in-time robot context. Does not execute robot commands."""
import ast
from pathlib import Path

READ_TOOL = 'read_robot_file'

def lifecycle_schema(name,tree):
    """Recognize the inspected local lifecycle API without importing SDK code."""
    if name not in {'activate_skill','deactivate_skill'}:return None
    imports=any(isinstance(n,ast.ImportFrom) and n.module=='tools.util.skills' for n in tree.body)
    declared=any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME' for t in n.targets)
        and isinstance(n.value,ast.Constant) and n.value.value==name for n in tree.body)
    keys={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='arguments' and n.func.attr=='get'
        and n.args and isinstance(n.args[0],ast.Constant)}
    if not imports or not declared or keys!={'name','reason'}:return None
    return {'type':'object','properties':{'name':{'type':'string','minLength':1},'reason':{'type':'string'}},
        'required':['name'] if name=='deactivate_skill' else [],'additionalProperties':False}


def compact_catalogue(host):
    rows = []
    for name, path in sorted(host.tools.items()):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        schema = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCHEMA' for t in node.targets):
                try:
                    schema = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        if schema is None:schema=lifecycle_schema(name,tree)
        rows.append({'name': name, 'schema': schema,
                     'description': (ast.get_docstring(tree) or '')[:1200],
                     'source_path': str(path.relative_to(host.workspace).as_posix()),
                     'schema_requires_read': schema is None})
    rows.append({'name': READ_TOOL, 'effect': 'read_only_no_simulation_steps',
                 'schema': {'type': 'object', 'properties': {
                     'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
                     'limit': {'type': 'integer', 'minimum': 1, 'maximum': 16000}},
                     'required': ['path'], 'additionalProperties': False},
                 'description': 'Read robot tool source or skill instructions only when needed; never motion.'})
    return rows


def current_instructions(host, state):
    """Load baseline + active skill; expose other skills by path, not full text."""
    paths = ['AGENTS.md', 'tools/README.md', 'robot_config/joints.md']
    for name in [state.get('baseline'), state.get('active_skill')]:
        if name is None:
            continue
        if not isinstance(name, str) or name in {'.', '..'} or '/' in name or '\\' in name:
            raise ValueError('Invalid skill name in authoritative state')
        paths.append('skills/' + name + '/SKILL.md')
    instructions = []
    for relative in dict.fromkeys(paths):
        if (host.workspace / relative).is_file():
            # Host performs containment / symlink checks before any read.
            # Keep the current procedure intact: truncating it can create extra
            # retrieval turns or hide mandatory release / verification steps.
            row = host.read_text(relative, limit=32000 if relative.startswith('skills/') else 16000)
            if relative.startswith('skills/') and row['total_characters'] > len(row['text']):
                remainder = host.read_text(relative, offset=len(row['text']), limit=32000)
                row = {**row, 'text': row['text'] + remainder['text']}
            row['complete'] = len(row['text']) == row['total_characters']
            instructions.append(row)
    index = []
    for path in sorted((host.workspace / 'skills').glob('*/SKILL.md')):
        index.append({'name': path.parent.name, 'path': path.relative_to(host.workspace).as_posix()})
    return {'instructions': instructions, 'available_skills': index,
            'retrieval_rule': 'Entries marked complete contain the entire document. Do not request text already provided. Use read_robot_file only for missing instructions or schema; do not guess arguments.'}


def read_document(host, arguments):
    if not isinstance(arguments, dict) or not {'path'} <= arguments.keys() or arguments.keys() - {'path', 'offset', 'limit'}:
        raise ValueError('Invalid read_robot_file arguments')
    limit = arguments.get('limit', 16000)
    if type(limit) is not int or not 1 <= limit <= 16000:
        raise ValueError('Document slice must be 1..16000 characters')
    return host.read_text(arguments['path'], offset=arguments.get('offset', 0), limit=limit)
