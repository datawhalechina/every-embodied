"""Extract actual robot video pixels and create clearly labeled custom UI states."""
import hashlib
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def prepare():
    folder = ROOT / 'assets/screen'
    sequence = folder / 'sequence'
    sequence.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'assets/source/nebula.mp4'
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        raise RuntimeError('Copy the authorized robot nebula.mp4 to assets/source first')
    frames = []
    fps=video.get(cv2.CAP_PROP_FPS)
    for index in range(min(900,int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ok, frame = video.read()
        if not ok:
            break
        path = sequence / f'nebula_{index+1:04d}.png'
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
        frames.append(path.name)
    video.release()
    if len(frames) < 2:
        raise RuntimeError('Video sequence is empty')
    im = Image.open(sequence / frames[12 if len(frames)>12 else 0]).convert('RGB')
    im.save(folder / 'screen.png')
    width, height = im.size
    for state in ['hello', 'listening', 'thinking', 'battery_unknown']:
        frame = Image.new('RGB', (width, height), '#030a13')
        draw = ImageDraw.Draw(frame)
        draw.ellipse((width*.07,height*.07,width*.93,height*.93), outline='#1a91ba', width=6)
        if state == 'hello':
            for x in (.34, .66):
                draw.rounded_rectangle((width*(x-.045),height*.3,width*(x+.045),height*.57), radius=30, fill='#72f3fa')
            draw.arc((width*.36,height*.51,width*.64,height*.75), 10,170, fill='#72f3fa', width=10)
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
        font = ImageFont.truetype(str(font_path), int(width*.06)) if font_path.exists() else ImageFont.load_default()
        labels = {'hello':'你好 AURA', 'listening':'正在聆听', 'thinking':'思考中…', 'battery_unknown':'电量：未知'}
        draw.text((width/2,height*.79), labels[state], anchor='mm', font=font, fill='#a2e7ef')
        frame.save(folder / (state+'.png'))
    record = {'source': '/home/sunrise/auraos/assets/display/nebula.mp4',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'sequence_source': 'actual_vendor_video_pixels', 'native_size': [width,height],
        'frames': len(frames), 'fps':fps, 'offset_s':0,
        'custom_states': ['hello','listening','thinking','battery_unknown'],
        'note': 'Custom states are local UI, not original firmware or measured battery status.'}
    (folder / 'provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    prepare()
