(() => {
  let activeCam = "booth_overview";
  let lastState = null;
  let camTimer = null;

  const $ = (id) => document.getElementById(id);

  function showToast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove("show"), 2600);
  }

  async function postJson(url, body = {}) {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "请求失败");
      return data;
    } catch (e) {
      showToast(`操作失败: ${e.message}`);
      throw e;
    }
  }

  function refreshCameraFrame() {
    const img = $("liveFeed");
    if (!img) return;
    const nextUrl = `/api/camera/${activeCam}.jpg?t=${Date.now()}`;
    const loader = new Image();
    loader.onload = () => { img.src = nextUrl; };
    loader.src = nextUrl;
  }

  function startCameraLoop() {
    if (camTimer) clearInterval(camTimer);
    refreshCameraFrame();
    const period = (activeCam === "c931" || activeCam === "real_wrist") ? 1800 : 280;
    camTimer = setInterval(() => {
      if (document.hidden) return;
      const isBusy = lastState && lastState.robot && lastState.robot.state !== "idle";
      if (activeCam === "c931" || activeCam === "real_wrist" || isBusy || Date.now() % 2 === 0) {
        refreshCameraFrame();
      }
    }, period);
  }

  // Bottom navigation
  document.querySelectorAll("#bottomNav .nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll("#bottomNav .nav-item").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
    });
  });

  // Camera tabs
  document.querySelectorAll("#camTabs .cam-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeCam = btn.dataset.cam;
      document.querySelectorAll("#camTabs .cam-tab").forEach((b) => b.classList.toggle("active", b === btn));
      startCameraLoop();
    });
  });

  // Main action buttons
  $("btnWelcome").addEventListener("click", async () => {
    await postJson("/api/voice/welcome");
    showToast("Magic Box 正在迎宾问候...");
  });
  document.querySelectorAll("[data-exec-mode]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const mode = btn.dataset.execMode;
      const res = await postJson("/api/robot/config", { execution_mode: mode });
      showToast(`已切换模式: ${res.execution_mode}`);
    });
  });

  document.querySelectorAll("[data-strategy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const strategy = btn.dataset.strategy;
      const res = await postJson("/api/robot/config", { strategy });
      showToast(`已切换策略: ${res.strategy}`);
    });
  });

  $("btnScan").addEventListener("click", async () => {
    await postJson("/api/robot/scan");
    showToast("开始扫描桌面场景并语音报库存...");
  });
  $("btnScanInv").addEventListener("click", async () => {
    await postJson("/api/robot/scan");
    showToast("开始视觉盘点...");
  });
  $("btnDemo").addEventListener("click", async () => {
    await postJson("/api/robot/full_booth_show");
    showToast("已启动 TuntunClaw 五阶段展台全体验！");
  });
  $("btnRecycle").addEventListener("click", async () => {
    await postJson("/api/robot/recycle", {});
    showToast("机械臂正在回收取货区物资并播报...");
  });
  $("btnHome").addEventListener("click", async () => {
    await postJson("/api/robot/home");
    showToast("机械臂正在回安全位...");
  });
  $("btnRestock").addEventListener("click", async () => {
    await postJson("/api/inventory/restock");
    showToast("已一键补货并复位桌面物料！");
  });
  $("btnStop").addEventListener("click", async () => {
    await postJson("/api/robot/stop");
    showToast("已触发急停！");
  });

  // Magic Box voice controls
  $("btnSpeakInv").addEventListener("click", async () => {
    await postJson("/api/voice/inventory");
    showToast("Magic Box 正在播报当前物资余量...");
  });
  $("voicePreset").addEventListener("change", async (e) => {
    await postJson("/api/voice/config", { voice_preset: e.target.value });
    showToast("音色已切换，后台正在预热清亮语音缓存...");
  });
  $("volSlider").addEventListener("input", (e) => {
    $("volVal").textContent = e.target.value;
  });
  $("volSlider").addEventListener("change", async (e) => {
    await postJson("/api/voice/config", { volume: Number(e.target.value) });
    showToast(`Magic Box 喇叭音量已调至 ${e.target.value}%`);
  });

  // QR Modal
  $("openQrBtn").addEventListener("click", () => $("qrModal").classList.remove("hidden"));
  $("closeQrBtn").addEventListener("click", () => $("qrModal").classList.add("hidden"));
  $("qrModal").addEventListener("click", (e) => {
    if (e.target === $("qrModal")) $("qrModal").classList.add("hidden");
  });

  // Add custom replenishment item
  $("addRepForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("repName").value.trim();
    const qty = parseInt($("repQty").value || "2", 10);
    const unit = $("repUnit").value.trim() || "件";
    if (!name) return;
    await postJson("/api/replenishment/add", { item_name: name, qty, unit });
    $("repName").value = "";
    showToast(`已加入补货清单: ${name}`);
  });

  // Copy replenishment checklist
  $("btnCopyList").addEventListener("click", async () => {
    if (!lastState) return;
    const pending = lastState.replenishment.filter((r) => r.status !== "done");
    const lines = [
      `【TuntunClaw 囤囤钳 · 展台补货清单】(${lastState.server_time})`,
      ...pending.map((r, i) => `${i + 1}. ${r.item_name} × ${r.suggested_qty}${r.unit} [${r.status === "ordered" ? "已下单" : "待补货"}] - ${r.reason}`),
    ];
    const txt = pending.length ? lines.join("\n") : "【TuntunClaw 囤囤钳】当前所有展台物料库存充足，无需补货。";
    try {
      await navigator.clipboard.writeText(txt);
      showToast("已复制补货清单，可直接粘贴发微信！");
    } catch {
      showToast(txt);
    }
  });

  // Real hardware probe
  $("btnHwConnect").addEventListener("click", async () => {
    showToast("正在探测真机 myCobot 280 Pi...");
    const res = await postJson("/api/hardware/connect", {
      host: $("hwHost").value.trim(),
      user: $("hwUser").value.trim(),
      password: $("hwPass").value,
    });
    if (res.connected) {
      showToast("真机 myCobot 280 Pi 连接成功！");
    } else {
      showToast(`未连上真机: ${res.error || "请检查 Wi-Fi/IP"}`);
    }
  });

  function renderState(st) {
    lastState = st;
    const r = st.robot || {};
    const v = st.voice || {};
    const sum = st.summary || {};

    const stateLabel = {
      idle: "待命中",
      scanning: "视觉扫描中",
      picking: "正在取送",
      recycling: "正在回收",
      moving: "运动中",
      jogging: "单轴微调",
      homing: "回安全位",
      real_mirror: "真实-仿真双同步",
      stopped: "已急停",
    }[r.state] || r.state;
    const badge = $("robotStateBadge");
    badge.textContent = stateLabel;
    badge.className = `badge ${r.state === "stopped" ? "badge-red" : r.state === "idle" ? "badge-teal" : "badge-amber"}`;

    document.querySelectorAll("[data-exec-mode]").forEach((b) =>
      b.classList.toggle("active", b.dataset.execMode === (st.execution_mode || "simulation"))
    );
    document.querySelectorAll("[data-strategy]").forEach((b) =>
      b.classList.toggle("active", b.dataset.strategy === (st.strategy || r.strategy || "servo_closed_loop"))
    );

    $("phaseText").textContent = r.phase || "就绪";
    const tcp = r.tcp_mm || [0, 0, 0];
    $("tcpText").textContent = `TCP (${tcp[0] >= 0 ? "+" : ""}${tcp[0]}, ${tcp[1] >= 0 ? "+" : ""}${tcp[1]}, ${tcp[2]}) mm`;
    $("progressBar").style.width = `${r.progress || 0}%`;
    $("gripPct").textContent = r.gripper_open_pct ?? 100;
    const trayItems = r.tray_items || [];
    $("trayCountText").textContent = `取货区待回收 ${trayItems.length} 件`;

    // Voice panel sync
    const vb = $("voiceBadge");
    if (v.daemon_online) {
      vb.textContent = `在线 · ${v.last_latency_ms ? v.last_latency_ms + "ms" : "已预热" + (v.cached_clips || 0) + "条"}`;
      vb.className = "badge badge-teal";
    } else {
      vb.textContent = "等待MagicBox";
      vb.className = "badge badge-muted";
    }
    if (v.last_text) {
      $("voiceBubble").textContent = `🔊 “${v.last_text}”`;
    }
    if (v.voice_preset && $("voicePreset").value !== v.voice_preset) {
      $("voicePreset").value = v.voice_preset;
    }
    if (typeof v.volume === "number" && document.activeElement !== $("volSlider")) {
      $("volSlider").value = v.volume;
      $("volVal").textContent = v.volume;
    }

    $("qrUrlText").textContent = st.access_url || window.location.origin;
    $("qrModalUrl").textContent = st.access_url || window.location.origin;

    const hw = st.hardware || {};
    const hwBadge = $("hwBadge");
    hwBadge.textContent = hw.connected ? `已连真机 ${hw.host}` : "孪生模式";
    hwBadge.className = `badge ${hw.connected ? "badge-teal" : "badge-muted"}`;

    $("kpiStock").textContent = sum.total_stock ?? 0;
    $("kpiDelivered").textContent = sum.total_delivered ?? 0;
    $("kpiLow").textContent = sum.low_stock_types ?? 0;
    $("kpiPending").textContent = sum.pending_replenishment ?? 0;

    const navBadge = $("navRepBadge");
    if ((sum.pending_replenishment || 0) > 0) {
      navBadge.textContent = sum.pending_replenishment;
      navBadge.classList.remove("hidden");
    } else {
      navBadge.classList.add("hidden");
    }

    // Quick pick grid
    const qp = $("quickPickGrid");
    qp.innerHTML = (st.inventory || []).map((item) => `
      <button class="pick-tile ${item.low_stock ? "low" : ""}" data-pick="${item.key}" ${item.quantity <= 0 ? "disabled" : ""}>
        <div class="pick-tile-top">
          <span class="pick-name"><i class="swatch" style="background:${item.color_hex}"></i>${item.name}</span>
          <span class="pick-qty">${item.quantity}${item.unit}</span>
        </div>
        <span class="pick-sub">${item.quantity > 0 ? "点击触发取送 + 分阶段语音" : "已空 · 可回收或补货"} ${item.low_stock ? "· ⚠️待补货" : ""}</span>
      </button>
    `).join("");
    qp.querySelectorAll("[data-pick]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const key = btn.dataset.pick;
        await postJson("/api/robot/pick", { item_key: key });
        showToast(`囤囤钳开始取送: ${key}`);
      });
    });

    // Joint grid
    const jg = $("jointGrid");
    const joints = r.joints_deg || [0, 0, 0, 0, 0, 0];
    jg.innerHTML = joints.map((deg, idx) => `
      <div class="joint-row">
        <div>
          <div class="joint-label">J${idx + 1}</div>
          <div class="joint-val">${deg >= 0 ? "+" : ""}${deg.toFixed(1)}°</div>
        </div>
        <div class="jog-btns">
          <button class="jog-btn" data-jog="${idx}" data-delta="-10">-</button>
          <button class="jog-btn" data-jog="${idx}" data-delta="10">+</button>
        </div>
      </div>
    `).join("");
    jg.querySelectorAll("[data-jog]").forEach((btn) => {
      btn.addEventListener("click", () => {
        postJson("/api/robot/jog", { joint_idx: Number(btn.dataset.jog), delta_deg: Number(btn.dataset.delta) });
      });
    });

    // Inventory list
    const il = $("inventoryList");
    il.innerHTML = (st.inventory || []).map((item) => `
      <div class="inv-row ${item.low_stock ? "low" : ""}">
        <div>
          <div class="inv-title"><i class="swatch" style="background:${item.color_hex}"></i>${item.name}
            <span class="badge ${item.low_stock ? "badge-red" : "badge-teal"}">${item.low_stock ? "需补货" : "库存充足"}</span>
          </div>
          <div class="inv-meta">${item.category} · 当前在取货区 ${item.delivered}${item.unit} · 更新 ${item.updated_at}</div>
        </div>
        <div><small class="mini-note">剩余</small><div class="inv-num">${item.quantity}<span style="font-size:14px">${item.unit}</span></div></div>
        <div class="inv-th">
          <span>阈值≤</span>
          <input type="number" min="0" max="20" value="${item.threshold}" data-th="${item.key}" />
        </div>
        <button class="btn btn-primary btn-sm" data-inv-pick="${item.key}" ${item.quantity <= 0 ? "disabled" : ""}>抓取递送</button>
      </div>
    `).join("");
    il.querySelectorAll("[data-inv-pick]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await postJson("/api/robot/pick", { item_key: btn.dataset.invPick });
        showToast("已下发抓取递送与语音指令！");
      });
    });
    il.querySelectorAll("[data-th]").forEach((inp) => {
      inp.addEventListener("change", async () => {
        await postJson(`/api/inventory/${inp.dataset.th}/threshold`, { threshold: Number(inp.value) });
        showToast("已更新补货预警阈值");
      });
    });

    // Replenishment checklist
    const rl = $("replenishList");
    const reps = st.replenishment || [];
    if (!reps.length) {
      rl.innerHTML = `<div class="sub-hint" style="text-align:center;padding:24px;">当前所有展台物料高于安全阈值，暂无待补货项目。</div>`;
    } else {
      rl.innerHTML = reps.map((rep) => `
        <div class="rep-item ${rep.status}">
          <div>
            <div class="rep-title">${rep.item_name} × ${rep.suggested_qty}${rep.unit}
              <span class="badge ${rep.status === "pending" ? "badge-red" : rep.status === "ordered" ? "badge-amber" : "badge-teal"}">
                ${rep.status === "pending" ? "待补货" : rep.status === "ordered" ? "已下单" : "已完成"}
              </span>
            </div>
            <div class="rep-reason">${rep.reason} · ${rep.created_at}</div>
          </div>
          <div class="rep-actions">
            ${rep.status === "pending" ? `<button class="btn btn-soft btn-sm" data-rep="${rep.id}" data-st="ordered">标记下单</button>` : ""}
            ${rep.status !== "done" ? `<button class="btn btn-primary btn-sm" data-rep="${rep.id}" data-st="done">已补货</button>` : `<button class="btn btn-soft btn-sm" data-rep="${rep.id}" data-st="pending">重新打开</button>`}
          </div>
        </div>
      `).join("");
      rl.querySelectorAll("[data-rep]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await postJson(`/api/replenishment/${btn.dataset.rep}/status`, { status: btn.dataset.st });
          showToast("补货清单状态已同步！");
        });
      });
    }

    // Event log
    const el = $("eventList");
    el.innerHTML = (st.events || []).slice(0, 14).map((ev) => `
      <li class="event-item ${ev.event_type}">
        <b>${ev.title}</b>
        <span>${ev.detail} · ${ev.created_at.slice(11)}</span>
      </li>
    `).join("");
  }

  function connectSSE() {
    const es = new EventSource("/api/events");
    es.onopen = () => {
      $("connPill").className = "pill pill-ok";
      $("connText").textContent = "实时同步";
    };
    es.onmessage = (ev) => {
      try {
        renderState(JSON.parse(ev.data));
      } catch (e) {
        console.error(e);
      }
    };
    es.onerror = () => {
      $("connPill").className = "pill pill-wait";
      $("connText").textContent = "重连中...";
    };
  }

  connectSSE();
  startCameraLoop();
})();
