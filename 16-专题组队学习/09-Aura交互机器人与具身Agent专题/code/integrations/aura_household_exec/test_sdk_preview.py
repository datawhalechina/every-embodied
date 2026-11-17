import ast,unittest
from pathlib import Path
from .sdk_preview import patched_sources

ROOT=Path(__file__).resolve().parents[2]/'tests/fixtures'

class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_tool=(ROOT/'tool-reference.py').read_text(encoding='utf-8')
        cls.original_media=(ROOT/'media-reference.py').read_text(encoding='utf-8')
        cls.tool,cls.media=patched_sources(cls.original_tool,cls.original_media)

    def preview_selection(self,source,name,arguments,result):
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='is_plan_preview' for t in n.targets))
        return eval(compile(ast.Expression(node.value),'sdk-selector','eval'),{'name':name,'arguments':arguments,'result':result,'PLAN_TOOLS':{'plan_grasp_point_filter_rgbd_lite','plan_press_point'}})

    def test_actual_upstream_omission_and_plan_fix(self):
        result={'ok':True,'marked_image_url':'actual-signed-plan.png'}
        self.assertFalse(self.preview_selection(self.original_tool,'move_tracked_point',{'execution_mode':'plan'},result))
        self.assertTrue(self.preview_selection(self.tool,'move_tracked_point',{'execution_mode':'plan'},result))
        self.assertFalse(self.preview_selection(self.tool,'move_tracked_point',{'execution_mode':'exec'},result))
        self.assertFalse(self.preview_selection(self.tool,'move_tracked_point',{'execution_mode':'plan'},{**result,'ok':False}))

    def test_plan_media_is_not_mislabelled_as_clickable_camera(self):
        tree=ast.parse(self.media);keep=[]
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and (t.id.endswith('_TOOLS') or t.id.endswith('_ORDER')) for t in n.targets):keep.append(n)
            if isinstance(n,ast.FunctionDef) and n.name in {'_nested_value','_selection_policy'}:keep.append(n)
        namespace={'Any':object};exec(compile(ast.Module(body=keep,type_ignores=[]),'media-selector','exec'),namespace)
        self.assertEqual(namespace['_selection_policy']('move_tracked_point',{'ok':True,'marked_image_url':'signed.png'})[0],'planning_overlay')
        self.assertNotEqual(namespace['_selection_policy']('move_tracked_point',{'ok':True,'feed':'head'})[0],'planning_overlay')

    def test_unknown_revision_is_not_silently_patched(self):
        with self.assertRaises(ValueError):patched_sources(self.tool,self.media)

if __name__=='__main__':unittest.main()
