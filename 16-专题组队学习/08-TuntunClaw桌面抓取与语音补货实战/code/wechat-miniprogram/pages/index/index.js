const app = getApp();

Page({
  data: {
    connected: false,
    activeCam: "booth_overview",
    camUrl: "",
    cameras: [
      { id: "booth_overview", label: "3D仿真总览" },
      { id: "wrist_rgb", label: "腕部RGB" },
      { id: "overhead_top", label: "俯视识别" },
      { id: "magicbox_cam", label: "MagicBox" },
      { id: "c931", label: "C931实拍" }
    ],
    robot: { state: "idle", phase: "待命中", tcp_mm: [0, 0, 0], gripper_open_pct: 100 },
    inventory: [],
    replenishment: [],
    summary: { total_stock: 0, total_delivered: 0, pending_replenishment: 0 },
    serverInput: ""
  },

  onLoad() {
    this.setData({ serverInput: app.globalData.serverBase });
    this.refreshAll();
    this.connectSocket();
    this._timer = setInterval(() => {
      this.refreshCam();
      if (!this.data.connected) this.fetchState();
    }, 800);
  },

  onUnload() {
    if (this._timer) clearInterval(this._timer);
    if (this._ws) this._ws.close();
  },

  onPullDownRefresh() {
    this.fetchState(() => wx.stopPullDownRefresh());
  },

  refreshAll() {
    this.fetchState();
    this.refreshCam();
  },

  refreshCam() {
    const base = app.globalData.serverBase;
    this.setData({
      camUrl: `${base}/api/camera/${this.data.activeCam}.jpg?t=${Date.now()}`
    });
  },

  switchCam(e) {
    this.setData({ activeCam: e.currentTarget.dataset.cam }, () => this.refreshCam());
  },

  connectSocket() {
    try {
      if (this._ws) this._ws.close();
      const ws = wx.connectSocket({ url: app.globalData.wsBase });
      this._ws = ws;
      ws.onOpen(() => this.setData({ connected: true }));
      ws.onMessage((res) => {
        try {
          const st = JSON.parse(res.data);
          if (st.robot) this.applyState(st);
        } catch (err) {}
      });
      ws.onClose(() => this.setData({ connected: false }));
      ws.onError(() => this.setData({ connected: false }));
    } catch (e) {}
  },

  fetchState(cb) {
    wx.request({
      url: `${app.globalData.serverBase}/api/state`,
      method: "GET",
      success: (res) => {
        if (res.statusCode === 200 && res.data && res.data.robot) {
          this.setData({ connected: true });
          this.applyState(res.data);
        }
      },
      complete: () => cb && cb()
    });
  },

  applyState(st) {
    this.setData({
      robot: st.robot,
      inventory: st.inventory || [],
      replenishment: st.replenishment || [],
      summary: st.summary || {}
    });
  },

  post(path, data = {}, toast = "") {
    wx.request({
      url: `${app.globalData.serverBase}${path}`,
      method: "POST",
      data,
      header: { "content-type": "application/json" },
      success: () => {
        if (toast) wx.showToast({ title: toast, icon: "none" });
        setTimeout(() => this.fetchState(), 350);
      },
      fail: () => wx.showToast({ title: "连接服务失败", icon: "none" })
    });
  },

  doWelcome() { this.post("/api/voice/welcome", {}, "Magic Box 正在迎宾"); },
  doScan() { this.post("/api/robot/scan", {}, "开始视觉扫描"); },
  doDemo() { this.post("/api/robot/full_booth_show", {}, "启动5阶段全体验"); },
  doRecycle() { this.post("/api/robot/recycle", {}, "正在回收取货区物资"); },
  doRestock() { this.post("/api/inventory/restock", {}, "已补货复位"); },
  doStop() { this.post("/api/robot/stop", {}, "已触发急停"); },
  doPick(e) {
    const key = e.currentTarget.dataset.key;
    this.post("/api/robot/pick", { item_key: key }, `开始抓取 ${key}`);
  },
  toggleRep(e) {
    const { id, next } = e.currentTarget.dataset;
    this.post(`/api/replenishment/${id}/status`, { status: next }, "清单状态已更新");
  },
  onServerInput(e) { this.setData({ serverInput: e.detail.value }); },
  saveServer() {
    const v = (this.data.serverInput || "").trim().replace(/\/$/, "");
    if (!v) return;
    app.globalData.serverBase = v;
    app.globalData.wsBase = v.replace(/^http/, "ws") + "/ws/live";
    wx.setStorageSync("tuntun_server_base", v);
    this.connectSocket();
    this.refreshAll();
    wx.showToast({ title: "服务地址已更新", icon: "success" });
  },
  openWebView() {
    wx.navigateTo({ url: `/pages/webview/webview?url=${encodeURIComponent(app.globalData.serverBase)}` });
  }
});
