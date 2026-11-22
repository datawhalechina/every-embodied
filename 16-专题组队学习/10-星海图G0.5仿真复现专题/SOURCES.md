# 代码和素材来源

## 双手弹琴

沿用仓库内的[双手弹琴原教程](../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/README.md)与其原始视频、乐曲署名、仿真回放包和脚本。专题中的 17 张画面由原章节的 `piano-xiaohongshu.mp4`（演播室发布成片）与 `public-relocation-demo.mp4`（移琴技术演示）提取。

## 叠衣服与桌面整理

- [逐回合数据](data/fold/episodes.csv)保留原生结局、得分和控制步。正式阶段为 30 回合，调试阶段为 6 回合。[场景清单](data/fold/case-plan.json)记录布局编号和散列。
- [动作与状态](data/fold/fold-mixed-trace.json.gz)来自 `fold_clothes__random__g0__l0` 组合回合的执行记录。[首个候选样本](data/fold/proposal-example.json)保留原始动作、夹爪范围校验后的动作和模型输入状态。
- 录像源自对应仿真回合的三相机合成画面。三段叠衣采用同一初始布局，桌面片段展示固定布局上的部分成功回合。
- 策略与仿真版本见[运行章节](02-叠衣服与桌面整理.md)。控制器补丁针对上层控制器公开源码的指定提交，沿用其 [MIT 许可证](code/robodojo/LICENSE.upstream)。

## 万圣节收纳

- [原生回合指标](data/halloween/runs.json)来自实例 301 的三个终局记录及录制讲解。无原生思维链组含早期动作与后续回合的连续拼接。归档动作流只含序号和动作数值，复验参考包含首帧及 65 维本体状态。
- 每组三段剪辑取自对应运行路径的重放录像。[片段清单](data/media.json)保留原始剪辑时间和截图在片段内的时间。播放时的剪辑时长与机器运行时间分开记录。
- [RoboHarness](https://github.com/BinceQu/RoboHarness) 提供控制工具、评测器和机器人配置。[BEHAVIOR-1K](https://behavior.stanford.edu/) 提供场景、任务与原生指标。外部资产遵循其各自的发布条款。

## 素材定位

静态教程中的图片位于 `assets/frames`，专题新增短片位于 `assets/clips`，三段完整解说位于各章节链接的原始发布页。运行 `python3 reproduce.py` 可在网页里按片段查看全部画面。
