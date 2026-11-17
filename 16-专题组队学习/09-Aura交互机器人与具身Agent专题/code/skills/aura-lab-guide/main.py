"""Minimal official SkillContext example with only the declared say tool."""


async def run(context, task: str = '') -> str:
    key = task.strip().lower()
    if '标签' in key or 'label' in key:
        message = '标签识读：在本机控制台选择标签画质，展示印刷文字和路标，检查 OCR 文本、路标编号与画质提示。'
    elif '方向' in key or 'direction' in key:
        message = '方向预览：先运行离线策略演示，核对人物在右侧时给出右转、居中时保持、目标丢失时停止。'
    elif '人物' in key or 'person' in key:
        message = '人物感知：选择来人举手预设，站到画面中并举手，检查多帧确认后的来人和举手候选事件。'
    else:
        message = '可选择标签识读、人物感知、方向预览。当前向导只介绍步骤；实机运行结果以控制台报告为准。'
    await context.say(message)
    return 'explained'
