"""Aura receives a command without starting any process or model."""
import datetime,hashlib,json,re,uuid
from pathlib import Path


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def receive(text,output,*,source='typed_user'):
    if not isinstance(text,str) or not 1<=len(text.strip())<=500:raise ValueError('请输入1至500字指令')
    if source!='typed_user':raise ValueError('This preparation entry accepts typed requests; ASR is not connected')
    if re.search(r'不要|取消|停止|别捡',text):raise ValueError('不会将停止或取消指令作为新家务启动')
    if '垃圾' not in text or not any(w in text for w in ('捡','收拾','清理')):raise ValueError('此场景先接入捡垃圾任务')
    identity=uuid.uuid4().hex[:16];folder=Path(output)/identity
    folder.mkdir(parents=True,exist_ok=False)
    request={'id':identity,'command':text.strip(),'input_source':source,'task_name':'picking_up_trash',
             'phase':'prepared_not_started','scene':'house_double_floor_lower','instance_id':301,
             'execution_started':False,'model_calls':0,'physical_robot_access':False,
             'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(folder/'request.json',request)
    write_new(folder/'aura-state.json',{'request_id':identity,'phase':'prepared_not_started',
        'screen':'listening','reply':'收到捡垃圾请求，场景与Agent已准备，等待后续启动。','voice_source':'none_not_synthesized'})
    return request


def verify_native_result(result,*,task='picking_up_trash',instance=301):
    if result.get('task')!=task or result.get('instance_id')!=instance:raise ValueError('Native result belongs to another task')
    success=result.get('success') is True
    q=result.get('q_score',{}).get('final')
    if not isinstance(q,(int,float)) or isinstance(q,bool) or not 0<=q<=1:raise ValueError('Invalid native score')
    return {'phase':'completed' if success else 'incomplete','native_success':success,'q_score':q,
            'reply':'垃圾清理完成。' if success else '本次尚未完成全部垃圾清理。'}
