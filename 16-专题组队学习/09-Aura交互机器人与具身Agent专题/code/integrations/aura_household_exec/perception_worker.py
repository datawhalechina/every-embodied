"""CPU-only pixels/audio worker. No simulator, scene state, or actor coordinates."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root/'perception-packages'))
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
    os.environ['YOLO_CONFIG_DIR']=str(root/'perception-config')
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    # Known local official YOLO weight, never an untrusted incoming pickle.
    weights=root/'yolov8n-pose.pt';digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    original=torch.load
    def load(*args,**kwargs):kwargs.setdefault('weights_only',False);return original(*args,**kwargs)
    torch.load=load
    from ultralytics import YOLO
    detector=YOLO(str(weights));torch.load=original
    from faster_whisper import WhisperModel
    recognizer=WhisperModel(str(root/'models/whisper-small'),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    counter=0;last=None;last_stamp=None
    print(json.dumps({'ready':True,'device':'cpu','yolo_weight_sha256':digest,'audio_model':'faster-whisper-small','uses_actor_truth':False}),flush=True)
    def file(relative):
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Perception input outside owned run')
        return path
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('shutdown'):break
        started=time.monotonic()
        try:
            path=file(request['file'])
            if request['kind']=='audio':
                segments,info=recognizer.transcribe(str(path),language='zh',beam_size=5,condition_on_previous_text=False,vad_filter=False)
                rows=[{'text':s.text,'start_s':s.start,'end_s':s.end,'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob} for s in segments]
                confidence=math.exp(sum(r['avg_logprob'] for r in rows)/len(rows)) if rows else 0
                result={'text':''.join(r['text'] for r in rows),'confidence':confidence,'segments':rows,
                        'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':'actual_cpu_asr_on_waveform'}
            elif request['kind']=='image':
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Image cannot be decoded')
                detection=detector.predict(image,device='cpu',imgsz=640,conf=.45,classes=[0],verbose=False)[0]
                people=[]
                for box,score,joints in zip(detection.boxes.xyxy.tolist(),detection.boxes.conf.tolist(),detection.keypoints.data.tolist()):
                    x=(box[0]+box[2])/2/image.shape[1]
                    # Track only pixel continuity; never use a scene actor identifier.
                    if last is None or abs(x-last)>.35 or request['stamp']-(last_stamp or 0)>2:counter+=1
                    raised=any(joints[w][2]>.4 and joints[s][2]>.4 and joints[w][1]<joints[s][1]-8 for w,s in [(9,5),(10,6)])
                    people.append({'track_id':'visual-'+str(counter),'center_x':x,'confidence':float(score),'waving':bool(raised),'box':box,'joints':joints})
                if len(people)==1:last=people[0]['center_x'];last_stamp=request['stamp']
                result={'people':people,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'image_id':request['image_id'],'stamp':request['stamp'],'source':'actual_yolo_pose_camera_pixels','uses_actor_truth':False}
                if request.get('annotated'):cv2.imwrite(str(file(request['annotated'])),detection.plot())
            else:raise ValueError('Unknown perception modality')
            print(json.dumps({'request_id':request['request_id'],'ok':True,'elapsed_s':round(time.monotonic()-started,3),**result}),flush=True)
        except Exception as error:
            print(json.dumps({'request_id':request.get('request_id'),'ok':False,'error_type':type(error).__name__,'error':str(error)[:500]}),flush=True)

if __name__=='__main__':main()
