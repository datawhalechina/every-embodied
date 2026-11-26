App({
  globalData: {
    // 展台调试默认指向本机局域网服务；正式上线请替换为已备案 HTTPS 域名
    serverBase: "http://192.168.1.102:8090",
    wsBase: "ws://192.168.1.102:8090/ws/live"
  },
  onLaunch() {
    const saved = wx.getStorageSync("tuntun_server_base");
    if (saved) {
      this.globalData.serverBase = saved;
      this.globalData.wsBase = saved.replace(/^http/, "ws") + "/ws/live";
    }
  }
});
