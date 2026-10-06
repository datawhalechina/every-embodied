# 演奏视频与素材署名

## 两份成片

| 视频 | 内容 | 时长 | 压缩前 / 发布大小 |
| --- | --- | --- | --- |
| [音乐室成片](assets/piano-xiaohongshu.mp4) | 小红书发布版：R1 Pro 双手多指弹奏《我爱你，中国》，音乐室与大屏画面 | 34.588 秒 | 5.64 MB / 2.46 MB |
| [钢琴移位续弹](assets/piano-relocation-performance.mp4) | 钢琴移位后，G0.5 辅助双臂重新就位并继续弹奏；加快了移位过渡段 | 35.960 秒 | 13.92 MB / 6.49 MB |

两份视频均为 1920 × 1080、25 fps。发布版仅重新编码画面，保留成片时间轴、字幕和 AAC 音轨；音轨直接复制，没有再次压缩。MP4 使用 faststart，支持网页边加载边播放。封面分别取自第 5 秒和第 16 秒。

音乐室成片沿用演奏前的 3 次 G0.5 双臂准备推理；移琴成片来自已记录的连续演奏会话，移琴后使用 4 次新的 G0.5 推理辅助双臂就位，再交回多指演奏控制器。该会话结果见 [relocation-session-report.json](assets/relocation-session-report.json)。视频剪辑与压缩没有重新运行模型。公开练习的独立复现入口见 [教程](README.md)。

声音由仿真中实际琴键事件生成，没有混入歌手人声或播放原录音。画面中的歌词采用用户提供文本。机器人与灵巧手来源见 [SOURCES.md](SOURCES.md)；音乐室外观参考上游 Isaac Sim 资源，原始贴图和模型资源不随成片打包。

## 背景图片与字体

以下图片用于屏幕及背景合成，经过裁剪、亮度调整、平移和画面编排；署名与许可证单独保留，不以本仓库许可证替代：

| 素材 | 作者及来源 | 许可证 |
| --- | --- | --- |
| 天安门 | LuxTonnerre，[TiananmenGatePic1](https://commons.wikimedia.org/wiki/File:TiananmenGatePic1.jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| 长城 | Velatrix / Nicolas M. Perrault，[Great Wall of China July 2006](https://commons.wikimedia.org/wiki/File:Great_Wall_of_China_July_2006.JPG) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| 五星红旗 | Roc0ast3r，[Flag of China at the Museum of Flight in Seattle (2024) - 0543](https://commons.wikimedia.org/wiki/File:Flag_of_China_at_the_Museum_of_Flight_in_Seattle_(2024)_-_0543.jpg) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| 手写字体 | Long Cang、Ma Shan Zheng | SIL Open Font License 1.1；本目录不分发字体文件 |

## 压缩方式

保留原有分辨率、帧率和播放速度，使用 FFmpeg 的 H.264 编码：

```bash
ffmpeg -i input.mp4 -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p \
  -c:a copy -movflags +faststart output.mp4
```

此命令用于成片发布，不替代仿真录制。音乐作品、图片、模型及字体的权利分别归其权利人所有；复用时请保留署名并确认相应用途的授权。
