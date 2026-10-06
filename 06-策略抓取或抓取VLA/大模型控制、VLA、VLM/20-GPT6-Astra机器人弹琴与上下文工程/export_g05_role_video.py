"""Present recorded model-assisted preparation beside unmodified piano playback.

This adds evidence and labels, not a new model capability. The inset is explicitly
an earlier preparation replay, slowed four times and then frozen on its last frame.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(args):
    os.environ['IMAGEIO_FFMPEG_EXE'] = str(args.ffmpeg)
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    report_path = args.session / 'report.json'
    prep_path = args.preparation / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    prep = json.loads(prep_path.read_text(encoding='utf-8'))
    if (not report.get('playback_completed') or report.get('safety_stop')
            or report.get('g05_preparation_report_sha256') != digest(prep_path)
            or not prep.get('success') or prep.get('mode') != 'g05_assisted'
            or prep.get('safety_stop') or not prep.get('model_calls')
            or prep.get('model_induced_command_distance_rad_seconds', 0) <= 0):
        raise ValueError('Require completed playback and matching model-assisted preparation')
    events = json.loads((args.session / 'events.json').read_text(encoding='utf-8'))
    first = min(e['time_seconds'] for e in events if e['type'] == 'NoteOn')
    with np.load(args.observation, allow_pickle=False) as observation:
        head = observation['head_rgb']
        if head.shape != (3, 360, 640) or head.dtype != np.uint8:
            raise ValueError('Expected recorded RGB head-camera observation')
        head_image = Image.fromarray(head.transpose(1, 2, 0))
    prep_video = args.preparation / 'recovery.mp4'
    with imageio.get_reader(str(prep_video)) as reader:
        prep_fps = reader.get_meta_data()['fps']
        prep_frames = [Image.fromarray(frame) for frame in reader]
    if not prep_frames or prep_frames[0].size != (960, 720):
        raise ValueError('Expected 960x720 preparation footage for this camera crop')
    fonts = {size: ImageFont.truetype(str(args.font), size) for size in (15, 16, 17, 18, 20, 24)}
    colors = dict(bg='#15191B', text='#F4F6F5', muted='#BCC4C2', model='#82D7B8', music='#EDC776')
    args.out.mkdir(parents=True, exist_ok=False)
    writer = None
    try:
        with imageio.get_reader(str(args.video)) as reader:
            fps = reader.get_meta_data()['fps']
            start_frame = max(0, round((first - .8) * fps))
            cut = start_frame / fps
            writer = imageio.get_writer(str(args.out / 'silent.mp4'), fps=fps,
                codec='libx264', quality=8, pixelformat='yuv420p')
            count = 0
            for frame_index, frame in enumerate(reader):
                if frame_index < start_frame:
                    continue
                main = Image.fromarray(frame)
                if main.size != (960, 720):
                    raise ValueError('Expected the full 960x720 performance take')
                canvas = Image.new('RGB', (1280, 720), colors['bg'])
                canvas.paste(main, (0, 0))
                draw = ImageDraw.Draw(canvas)

                def text(x, y, value, size=17, color='text', width=288):
                    if draw.textlength(value, font=fonts[size]) > width:
                        raise ValueError(f'Caption overflows: {value}')
                    draw.text((x, y), value, font=fonts[size], fill=colors[color])

                t = count / fps
                inset_index = min(int(t * prep_fps / 4), len(prep_frames) - 1)
                inset = prep_frames[inset_index].crop((310, 95, 740, 407))
                canvas.paste(ImageOps.fit(inset, (288, 209)), (976, 91))
                text(976, 20, 'G0.5 × 双手钢琴', 24, 'model')
                text(976, 56, '右侧：演奏前准备记录', 17, 'muted')
                caption = ('双臂准备回放 · 4 倍慢放' if t < len(prep_frames) / prep_fps * 4
                           else '双臂准备已完成 · 末帧留档')
                text(976, 307, caption, 17, 'model')
                text(976, 342, '输入示例：头部相机，第 1 次调用', 16, 'muted')
                canvas.paste(head_image.resize((288, 162), Image.Resampling.LANCZOS), (976, 371))
                text(976, 548, f'准备阶段 · {len(prep["model_calls"])} 次模型推理', 20, 'model')
                text(976, 583, 'G0.5：提出双臂关节目标', 17)
                text(976, 611, 'IK：对齐与限幅，再交还演奏', 17)
                draw.line((976, 645, 1264, 645), fill='#454D4A', width=1)
                text(976, 658, 'G0.5 双臂准备 → 专用控制器演奏', 15, 'model')
                text(976, 684, '接力完成这段双手钢琴演示', 15, 'muted')
                draw.rectangle((0, 654, 960, 720), fill=colors['bg'])
                text(22, 662, 'R1 Pro + Shadow Hand  /  双手多指演奏', 20, 'music', width=916)
                text(22, 694, '主画面正常速度 · 曲谱指法 + IK · 音轨由实际按键事件生成', 16, 'muted', width=916)
                writer.append_data(np.asarray(canvas))
                if count in (round(fps), round(8 * fps)):
                    canvas.save(args.out / f'frame-{count:04d}.png')
                count += 1
            writer.close()
            writer = None
        subprocess.run([str(args.ffmpeg), '-nostdin', '-n', '-loglevel', 'error',
            '-i', str(args.out / 'silent.mp4'), '-ss', str(cut), '-i', str(args.video),
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
            '-b:a', '192k', '-af', 'apad', '-t', str(count / fps),
            '-movflags', '+faststart', str(args.out / 'demo.mp4')], check=True)
        evidence = dict(source_video_sha256=digest(args.video), source_report_sha256=digest(report_path),
            preparation_report_sha256=digest(prep_path), preparation_video_sha256=digest(prep_video),
            observation_sha256=digest(args.observation), removed_silent_lead_seconds=cut,
            first_physical_note_seconds=first-cut, duration_seconds=count/fps,
            model_call_indices=prep['model_calls'], model_role='recorded model-assisted arm preparation',
            preparation_inset_slowdown=4, inset_then_freezes=True, performance_speed=1,
            original_music_events_unchanged=True, new_model_invocations=0,
            original_performance_passed=report['performance_passed'],
            limits=['Presentation only; no new controller capability', 'No finger actions from G0.5',
                    'Not evidence of benefit over IK', 'Inset and main view are from different times'])
        (args.out / 'presentation-report.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps(evidence, indent=2))
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'preparation', 'observation', 'video', 'font', 'ffmpeg', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    render(parser.parse_args())
