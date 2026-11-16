"""Render portable recorded evidence separately from live model execution."""
from pathlib import Path
import hashlib
import html
import json
import math


def load_snapshot(path):
    path = Path(path)
    manifest = json.loads((path / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files_sha256'].items():
        actual = hashlib.sha256((path / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Recorded evidence changed: {relative}')
    return path, manifest


def table(headers, rows):
    from IPython.display import HTML, display
    def cells(values, tag):
        return ''.join(f'<{tag} style="padding:8px;border-bottom:1px solid #dce3ec">{html.escape(str(v))}</{tag}>' for v in values)
    display(HTML('<table style="border-collapse:collapse"><tr>' + cells(headers, 'th') + '</tr>' +
                 ''.join('<tr>' + cells(row, 'td') + '</tr>' for row in rows) + '</table>'))


def metrics(path):
    rows = [json.loads(line) for line in (Path(path) / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if [r['step'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Recorded update sequence is incomplete or duplicated.')
    if not all(math.isfinite(row['loss']) for row in rows):
        raise ValueError('Recorded loss contains non-finite values.')
    return rows


def training_summary(path):
    path, manifest = load_snapshot(path)
    config = json.loads((path / 'formal_config.json').read_text())
    info = json.loads((path / 'dataset_info.json').read_text())
    rows = metrics(path)
    measured = json.loads((path / ('trained_model.json' if (path / 'trained_model.json').is_file() else 'training_measurements.json')).read_text())
    values = [('任务', config['task']), ('数据回合／帧数', f"{info['total_episodes']} / {info['total_frames']}"),
              ('真实已记录更新／目标', f"{len(rows)} / {config['steps']}"), ('实际批量／梯度累积／有效批量', '1 / 1 / 1'),
              ('学习率或峰值学习率', config['learning_rate']), ('可训练参数', measured.get('trainable_parameters', '尚未记录')),
              ('最后记录损失', rows[-1]['loss']), ('正式训练状态', '完成' if len(rows) == config['steps'] else '仍在进行，尚无最终权重成绩'),
              ('原始证据目录', manifest['source_run_root'])]
    if measured.get('duration_s') is not None:
        values.append(('实测训练阶段耗时／小时', round(measured['duration_s'] / 3600, 3)))
    if measured.get('checkpoint_index') is not None:
        values.append(('原生检查点编号（从零开始）', measured['checkpoint_index']))
    peak = measured.get('device_memory_stats', {}).get('peak_bytes_in_use')
    if peak is not None:
        values.append(('框架记录的峰值活动设备内存／二进制吉字节', round(peak / 2**30, 3)))
    table(['记录项', '实测或配置'], values)
    print('样本更新次数／数据帧数：', round(len(rows) * config['batch_size'] / info['total_frames'], 4))
    print('该比值用于理解数据使用规模；随机有放回采样不保证每个样本都已遍历。')
    model = path / 'trained_model.json'
    if model.is_file():
        record = json.loads(model.read_text())
        print('本次权重内容指纹：', record['checkpoint_sha256'])
        print('参数更新前后指纹是否不同：', record.get('parameter_digest_before') != record.get('parameter_digest_after'))
    return {'updates': len(rows), 'target': config['steps']}


def plot_training(path):
    import matplotlib.pyplot as plt
    rows = metrics(path)
    from matplotlib import font_manager
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font_path)).get_name(), 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(9, 3.6))
    steps = [r['step'] for r in rows]
    losses = [r['loss'] for r in rows]
    ax.plot(steps, losses, color='#a6b5ca', linewidth=.5, label='逐步真实损失')
    window = min(100, len(rows))
    rolling = []
    total = 0
    for i, value in enumerate(losses):
        total += value
        if i >= window:
            total -= losses[i-window]
        rolling.append(total / min(i+1, window))
    ax.plot(steps, rolling, color='#245cb3', linewidth=1.5, label=f'过去至多 {window} 步均值')
    ax.set(xlabel='真实梯度更新步数', ylabel='训练损失', title='训练损失与滑动均值；不代表闭环成功率')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); plt.show()


def evaluation_rows(path, pattern):
    path = Path(path)
    attempts = [p for p in path.glob(pattern) if (p / 'summary.json').is_file()]
    config = json.loads((path / 'formal_config.json').read_text())
    if config.get('policy_loader_contract'):
        attempts = [p for p in attempts if json.loads((p / 'summary.json').read_text()).get('loader_contract') == config['policy_loader_contract']]
    attempts = [p for p in attempts if json.loads((p / 'summary.json').read_text()).get('evaluation_clock', 'official_async') == config.get('evaluation_clock', 'official_async')]
    if not attempts:
        return None, []
    directory = max(attempts, key=lambda p: (p / 'summary.json').stat().st_mtime)
    summary = json.loads((directory / 'summary.json').read_text())
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines() if line.strip()]
    if [r['seed'] for r in rows] != summary['seeds'] or len(rows) != summary['episodes']:
        raise ValueError('Evaluation denominator or seed identity is inconsistent.')
    return summary, rows


def compare_policies(path):
    base, base_rows = evaluation_rows(path, 'baseline_*')
    trained, trained_rows = evaluation_rows(path, 'eval_*')
    records = []
    config = json.loads((Path(path) / 'formal_config.json').read_text())
    base_label = '官方起始任务权重' if config['task'] == 'water_plant' else '原始基座'
    for label, summary in [(base_label, base), ('本次训练后权重', trained)]:
        records.append([label, f"{summary['successes']} / {summary['episodes']}" if summary else '尚未完成',
                        f"{summary['success_rate']:.1%}" if summary else '不填写',
                        summary['checkpoint_sha256'][:16] if summary else '尚未产生最终评估'])
    table(['策略', '成功回合／完整分母', '成功率', '权重指纹前缀'], records)
    if config.get('policy_loader_contract') and base and trained:
        table(['策略', '适配张量数', '允许静默丢弃参数', '计时前预热／随机数恢复'],
              [[label, summary['model_load_audit']['adapter_tensor_count'],
                summary['model_load_audit']['remove_extra_params'],
                f"{summary['warmup_policy']} / {summary['warmup']['rng_restored']}"]
               for label, summary in [(base_label, base), ('本次训练后权重', trained)]])
    if not base or not trained:
        print('正式对照尚未齐全；不使用短程验收或历史模型成绩补齐。')
        return
    if base['seeds'] != trained['seeds']:
        raise ValueError('Paired comparison requires identical ordered seeds.')
    paired = [[a['seed'], '成功' if a['success'] else '失败', '成功' if b['success'] else '失败',
               '改善' if not a['success'] and b['success'] else '退步' if a['success'] and not b['success'] else '相同']
              for a, b in zip(base_rows, trained_rows)]
    table(['种子', '基座', '训练后', '同种子变化'], paired)
    print('总体成功率差：', f"{100*(trained['success_rate']-base['success_rate']):+.1f} 个百分点")
    if config.get('evaluation_clock') == 'pause_while_waiting':
        clock_rows = []
        for seed in base['seeds']:
            values = [seed]
            for summary in [base, trained]:
                audit_path = Path(path) / Path(summary['directory']).name / f'seed_{seed}' / 'clock_audit.json'
                audit = json.loads(audit_path.read_text())
                if audit['evaluation_clock'] != 'pause_while_waiting':
                    raise ValueError('Clock audit belongs to another deployment condition.')
                values.extend([audit['physics_steps'], round(audit['inference_wait_seconds'], 3), round(audit['wall_seconds'], 3)])
            clock_rows.append(values)
        table(['种子', '起始物理步数', '起始等待休眠秒', '起始墙钟秒', '训练后物理步数', '训练后等待休眠秒', '训练后墙钟秒'], clock_rows)
        print('等待列为缓冲区为空时休眠时间之和，未包含循环处理开销；墙钟列包含完整回合耗时。视频按物理仿真步播放，此条件不代表实时推理性能。')
    print('本结果只适用于记录的单任务、种子、划分及配方；相同成功率不代表每个回合行为相同。')


def plot_deployment_conditions(official_path, paused_path):
    """Compare complete, separately labeled deployment conditions with identical weights."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font_path)).get_name(), 'DejaVu Sans']
    groups = []
    for path, mode in [(official_path, 'official_async'), (paused_path, 'pause_while_waiting')]:
        summaries = [evaluation_rows(path, pattern)[0] for pattern in ['baseline_*', 'eval_*']]
        for summary in summaries:
            if not summary or summary['episodes'] != 20 or summary['seeds'] != list(range(100, 120)):
                raise ValueError('Deployment comparison requires both complete fixed-seed evaluations.')
            if summary.get('evaluation_clock', 'official_async') != mode:
                raise ValueError('Deployment condition is mislabeled.')
        groups.append(summaries)
    if any(groups[0][i]['checkpoint_sha256'] != groups[1][i]['checkpoint_sha256'] for i in range(2)):
        raise ValueError('Deployment conditions must use identical checkpoint contents.')
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (label, color) in enumerate([('官方起始任务权重', '#2d68ac'), ('本次训练后权重', '#df8b3e')]):
        positions = [x + (i - .5) * .32 for x in range(2)]
        values = [group[i]['success_rate'] * 100 for group in groups]
        bars = ax.bar(positions, values, width=.30, label=label, color=color)
        ax.bar_label(bars, labels=[f"{group[i]['successes']}/20" for group in groups], padding=3)
    ax.set(xticks=[0, 1], xticklabels=['官方异步时序', '等待时暂停仿真（独立条件）'],
           ylabel='固定 20 种子成功率（%）', ylim=(0, 108), title='相同权重在两种部署条件下的完整评估')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout(); plt.show()
    print('每组各 20 个固定种子；暂停仿真没有提高设备推理速度，不能替代官方异步成绩。')


def show_recorded_videos(path):
    from IPython.display import Markdown, Video, display
    path, manifest = load_snapshot(path)
    for video in manifest['selected_videos']:
        label = ('官方起始任务权重' if manifest['benchmark'] == 'dexjoco' else '原始基座') if video['variant'] == 'base' else '本次训练后权重'
        display(Markdown(f"**{label} · 种子 {video['seed']} · {'成功' if video['success'] else '失败'}**"))
        display(Video(str(path / video['path']), embed=True, width=560))
    if not manifest['selected_videos']:
        print('正式评估视频尚未产生；未以其他权重的视频代替。')
    for video in manifest.get('historical_selected_videos', []):
        caption = video.get('caption', f"历史评估 · 种子 {video['seed']} · 成功；该轮官方基线加载有缺陷，不计入当前修正后的主对照")
        display(Markdown(f"**{caption}**"))
        display(Video(str(path / video['path']), embed=True, width=560))


def show_execution_validation(path):
    from IPython.display import Markdown, Video, display
    root = Path(path) / 'execution_validation'
    if not root.is_dir():
        return
    display(Markdown('**独立的整本执行验收：20 步训练、1 回合评估。下方仅证明执行链路，不计入正式 30000 步实验。**'))
    status = json.loads((root / 'evaluate_status.json').read_text())['result']
    table(['训练步数', '评估分母', '成功回合', '用途'], [[20, status['episodes'], status['successes'], '整本执行验收']])
    if (root / 'front.mp4').is_file():
        display(Video(str(root / 'front.mp4'), embed=True, width=560))


def record_live_run(config):
    """Copy this run's lightweight evidence and representative videos for display."""
    import shutil
    root = Path(config['run_root'])
    destination = root / 'course_evidence'
    destination.mkdir(exist_ok=True)
    for source in root.glob('*.json'):
        shutil.copy2(source, destination / source.name)
    shutil.copy2(root / 'metrics.jsonl', destination / 'metrics.jsonl')
    shutil.copy2(Path(config['dataset_root']) / 'meta/info.json', destination / 'dataset_info.json')
    (destination / 'formal_config.json').write_text(json.dumps(config, indent=2) + '\n')
    selected = []
    for variant, pattern in [('base', 'baseline_*'), ('trained', 'eval_*')]:
        attempts = [p for p in root.glob(pattern) if p.is_dir() and (p / 'summary.json').is_file()]
        attempts = [p for p in attempts if json.loads((p / 'summary.json').read_text()).get('evaluation_clock', 'official_async') == config.get('evaluation_clock', 'official_async')]
        if config.get('policy_loader_contract'):
            attempts = [p for p in attempts if json.loads((p / 'summary.json').read_text()).get('loader_contract') == config['policy_loader_contract']]
        if not attempts:
            continue
        attempt = max(attempts, key=lambda p: (p / 'summary.json').stat().st_mtime)
        dest = destination / attempt.name
        dest.mkdir(exist_ok=True)
        for filename in ['summary.json', 'results.jsonl', 'evaluation_config.json']:
            if (attempt / filename).is_file():
                shutil.copy2(attempt / filename, dest / filename)
        summary = json.loads((attempt / 'summary.json').read_text())
        rows = [json.loads(line) for line in (attempt / 'results.jsonl').read_text().splitlines()]
        seeds = [next(row['seed'] for row in rows if row['success'] == outcome)
                 for outcome in [True, False] if any(row['success'] == outcome for row in rows)]
        for seed in seeds:
            video = next((attempt / f'seed_{seed}').rglob('*.mp4'))
            target = destination / 'videos' / f'{variant}_seed_{seed}.mp4'
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(video, target)
            selected.append({'variant': variant, 'seed': seed, 'success': next(r['success'] for r in rows if r['seed'] == seed),
                             'path': str(target.relative_to(destination)), 'checkpoint_sha256': summary['checkpoint_sha256']})
    manifest = {'benchmark': config['benchmark'], 'source_run_root': str(root),
                'formal_target_steps': config['steps'], 'selected_videos': selected, 'files_sha256': {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob('*') if p.is_file() and p.name != 'snapshot_manifest.json'}}
    (destination / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return destination
