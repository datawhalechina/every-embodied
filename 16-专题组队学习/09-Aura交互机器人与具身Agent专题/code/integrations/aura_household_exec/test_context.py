import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from .context import compact_catalogue,current_instructions,read_document,READ_TOOL


class Host:
    def __init__(self, root):
        self.workspace=Path(root); self.tools={}; self.reads=[]
    def read_text(self, relative, offset=0, limit=16000):
        path=(self.workspace/relative).resolve()
        if not path.is_relative_to(self.workspace.resolve()):raise ValueError('escape')
        self.reads.append(relative);text=path.read_text(encoding='utf-8')
        return {'path':relative,'offset':offset,'total_characters':len(text),'text':text[offset:offset+limit]}
    def write(self,path,text):
        p=self.workspace/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8');return p


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.host=Host(self.folder.name)
    def test_catalogue_keeps_schema_but_not_tool_implementation(self):
        self.host.tools['move']=self.host.write('tools/move.py','"""Move with feedback."""\nSCHEMA={"x":"number"}\nSECRET_IMPLEMENTATION="not in catalogue"')
        row=compact_catalogue(self.host)[0]
        self.assertEqual(row['schema'],{'x':'number'});self.assertNotIn('SECRET_IMPLEMENTATION',str(row))
    def test_dynamic_schema_explicitly_requires_source_read(self):
        self.host.tools['move']=self.host.write('tools/move.py','SCHEMA=dict(x="number")')
        self.assertTrue(compact_catalogue(self.host)[0]['schema_requires_read'])
    def test_inspected_lifecycle_contract_avoids_repeated_source_reads(self):
        source='from tools.util.skills import main\nNAME="deactivate_skill"\ndef deactivate(arguments):\n return arguments.get("name"), arguments.get("reason", "")\n'
        self.host.tools['deactivate_skill']=self.host.write('tools/deactivate_skill.py',source)
        row=compact_catalogue(self.host)[0]
        self.assertEqual(row['schema']['required'],['name']);self.assertFalse(row['schema_requires_read'])
    def test_changed_lifecycle_contract_falls_back_to_source_read(self):
        source='from tools.util.skills import main\nNAME="activate_skill"\ndef activate(arguments):\n return arguments.get("name"), arguments.get("reason"), arguments.get("force")\n'
        self.host.tools['activate_skill']=self.host.write('tools/activate_skill.py',source)
        self.assertTrue(compact_catalogue(self.host)[0]['schema_requires_read'])
    def test_only_active_and_baseline_skill_text_is_loaded(self):
        for name in ['base','pick','place']:self.host.write('skills/'+name+'/SKILL.md',name+' full instructions')
        value=current_instructions(self.host,{'baseline':'base','active_skill':'pick'})
        self.assertNotIn('place full instructions',str(value));self.assertEqual(len(value['available_skills']),3)
        self.assertNotIn('skills/place/SKILL.md',self.host.reads)
    def test_skill_switch_changes_context(self):
        self.host.write('skills/place/SKILL.md','place instructions')
        self.assertIn('place instructions',str(current_instructions(self.host,{'active_skill':'place'})))
    def test_current_procedure_is_not_cut_at_16000(self):
        self.host.write('skills/place/SKILL.md','x'*36000+' release and verify')
        value=current_instructions(self.host,{'active_skill':'place'})
        self.assertIn('release and verify',value['instructions'][0]['text'])
        self.assertTrue(value['instructions'][0]['complete'])
    def test_invalid_skill_path_rejected(self):
        with self.assertRaises(ValueError):current_instructions(self.host,{'active_skill':'../auth'})
    def test_readonly_document_slice_and_bounds(self):
        self.host.write('tools/move.py','abcdefgh')
        self.assertEqual(read_document(self.host,{'path':'tools/move.py','offset':2,'limit':3})['text'],'cde')
        for args in [{'path':'../auth'}, {'path':'tools/move.py','limit':16001}, {'path':'tools/move.py','command':'move'}]:
            with self.assertRaises(ValueError):read_document(self.host,args)
    def test_opt_in_controller_retrieval_never_executes_robot_tool(self):
        from .controller import run
        host=self.host
        host.evidence=host.workspace/'evidence';host.evidence.mkdir()
        host.tools['move']=host.write('tools/move.py','SCHEMA={"x":"number"}')
        host.write('skills/active.json','{"active_skill":null}')
        host.write('perception/latest.json',json.dumps({'images':{k:{'path':k+'.png'} for k in ['head_rgb','left_wrist_rgb','right_wrist_rgb']}}))
        proof=host.write('qualification.json','{"passed":true}')
        exe=host.write('fake-cli','not executed')
        binding={'session_id':'s','episode_id':'e','sequence':3,'head_image_id':'img','proprioception':[], 'native_ticks':4}
        host.frame=lambda:(binding,[]);host.unresolved=lambda:[]
        host.execute=lambda *args: (_ for _ in ()).throw(AssertionError('Document retrieval executed physical tool'))
        prompts=[]
        class Client:
            returncode=0
            def __init__(self,cmd,**kwargs):self.cmd=cmd
            def communicate(self,text,timeout):
                prompts.append(json.loads(text));done=len(prompts)==2
                row={'summary':'read' if not done else 'stop','expected_check':'no motion','tool':'none' if done else READ_TOOL,
                     'arguments_json':'{}' if done else '{"path":"tools/move.py"}','sequence':3,'head_image_id':'img','done':done,'memory':'public'}
                Path(self.cmd[self.cmd.index('--output-last-message')+1]).write_text(json.dumps(row),encoding='utf-8')
                return '{"type":"turn.completed","usage":{"input_tokens":1}}',''
        with patch('subprocess.Popen',Client):
            result=run(host,'test only',exe,qualification=proof,max_turns=2,context_mode='compact')
        self.assertTrue(result['model_requests_finish'])
        self.assertEqual(prompts[1]['requested_document']['text'],'SCHEMA={"x":"number"}')

if __name__=='__main__':unittest.main()
