"""Summarize fixed local hardware snapshots; never contacts a device."""
import json
import math
from pathlib import Path


def read(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def celsius(value):
    try:
        number = float(value) / 1000
        return round(number, 2) if math.isfinite(number) and -100 <= number <= 200 else None
    except (ValueError, TypeError):
        return None


def summarize(artifacts):
    inventory = read(artifacts/'hardware-readonly-20260908.json')
    thermal = read(artifacts/'thermal-readonly-20260908.json')
    temperatures = []
    for name, values in thermal.get('thermal', {}).items():
        if not isinstance(values, dict):
            continue
        passive, critical = [], []
        for key, value in values.items():
            if key.startswith('trip_point_') and key.endswith('_type'):
                temp = celsius(values.get(key[:-5]+'_temp'))
                if temp is not None and value in ('passive', 'critical'):
                    (passive if value == 'passive' else critical).append(temp)
        reading = celsius(values.get('temp'))
        threshold = min(passive) if passive else None
        temperatures.append({'zone': name, 'type': values.get('type'), 'temperature_c': reading,
                             'passive_trip_c': threshold, 'critical_trip_c': min(critical) if critical else None,
                             'at_or_above_passive_trip': reading is not None and threshold is not None and reading >= threshold})
    sources = inventory.get('pulse_sources')
    inputs = []
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and str(source.get('name', '')).startswith('alsa_input.'):
                inputs.append({'mute': source.get('mute'), 'state': source.get('state'),
                               'note': '仅输入配置快照，不证明存在有效人声。'})
    return {'status': 'historical_snapshot' if inventory or thermal else 'no_snapshot',
            'live': False, 'inventory_sample_time': inventory.get('time'),
            'thermal_sample_time': thermal.get('time'), 'temperatures': temperatures, 'audio_inputs': inputs,
            'power_supply_nodes': len(inventory['power_supplies']) if isinstance(inventory.get('power_supplies'), dict) else None,
            'battery_percent': None, 'charging': None,
            'note': '仅固定本地历史文件；不是当前温度或联网检查。温度采样时间若为空则未记录，不拿文件修改时间冒充。电量百分比/充电仍未知。',
            'next_steps': ['现场确认开机、风扇和通风口；不要拆机或触碰内部热部件。',
                           '恢复连接后重新读取温度与阈值，验证相机桥实际帧率/CPU；历史高温未视为已解除。',
                           '音频须用有效人声验收；电量/充电须有厂家协议或可靠电源驱动，不能由电压猜测。']}
