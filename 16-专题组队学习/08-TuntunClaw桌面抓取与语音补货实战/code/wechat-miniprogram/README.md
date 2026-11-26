# 豚豚CLOUD · 微信小程序封装与实时显示说明

本目录（`wechat-miniprogram/`）是可直接导入 **微信开发者工具** 的原生微信小程序工程，同时内置两种接入模式：

1. **原生页面模式（`pages/index/index`，推荐展台实时显示）**
   - 直接通过 `wx.connectSocket`（WebSocket 实时推送）+ 定时刷新 JPEG 帧（`/api/camera/*.jpg`）在微信小程序内**零延迟实时显示**机械臂画面、6 轴遥测、物料库存和补货清单。
   - **为什么不用 MJPEG？** 微信小程序 `<image>` 组件不支持 `multipart/x-mixed-replace` 长流，改用带时间戳的 JPEG 帧刷新 + WebSocket 状态推送，在 iOS / Android 微信里均 100% 稳定流畅。
2. **H5 网页壳模式（`pages/webview/webview`）**
   - 直接把 `web/` 整个响应式网页小程序嵌进 `<web-view>`，适合已有备案 HTTPS 业务域名时零维护复用。

---

## 快速跑通步骤

### 场景 A：展台现场 / 局域网体验版（最省心，无需备案域名）
1. 在本机运行 `start_tuntuncloud.ps1`，确认终端打印的局域网地址（例如 `http://192.168.1.102:8090`）。
2. 打开 **微信开发者工具** → 导入本目录 `wechat-miniprogram/`（测试号或自有 AppID 均可）。
3. 顶部「详情 → 本地设置」勾选 **不校验合法域名、web-view、TLS 版本以及 HTTPS 证书**。
4. 点击 **预览** 生成微信二维码；手机连同一个展台 Wi-Fi，扫码进入后点右上角「··· → 打开调试」，即可在微信里**实时看画面、一键扫描、一键抓取和勾选补货清单**！

### 场景 B：正式版公开扫码（观众无需连展台 Wi-Fi）
微信小程序正式版强制要求 `HTTPS` / `WSS` + ICP 备案域名：
1. 用 frp / cpolar / Cloudflare Tunnel 将本地 `8090` 端口映射到你的备案 HTTPS 域名（如 `https://tuntun.yourdomain.com`）。
2. 在微信公众平台 → 开发管理 → 开发设置 → 服务器域名中配置：
   - `request` 合法域名：`https://tuntun.yourdomain.com`
   - `socket` 合法域名：`wss://tuntun.yourdomain.com`
   - 若使用 `web-view`，额外在「业务域名」中上传校验文件并添加该域名。
3. 修改 `app.js` 中的 `serverBase` 为该 HTTPS 域名，提交审核发布即可生成永久展台小程序码。
