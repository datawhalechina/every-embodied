/* Shared showcase model + a small, dependency-free view inside the ebook. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TeamLearning = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const icon = (name) => name === "play"
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7z" fill="currentColor"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-8 8M10 4H4v16h16v-6" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>';

  function validMedia(value) {
    return typeof value === "string" && value.length > 0 && !/^(?:javascript|data|file):/i.test(value)
      && !value.startsWith("//") && !value.includes("\\") && !value.split("/").includes("..")
      && (/^https:\/\//.test(value) || !value.includes(":"));
  }
  function validateData(data) {
    for (const key of ["groups", "topics", "cases"]) {
      if (!Array.isArray(data[key]) || !data[key].length) throw new Error(`Missing showcase ${key}`);
      if (new Set(data[key].map((row) => row.id)).size !== data[key].length) throw new Error(`Duplicate ${key} id`);
    }
    const topics = new Set(data.topics.map((t) => t.id));
    const groups = new Set(data.groups.map((g) => g.id));
    const cases = new Map(data.cases.map((c) => [c.id, c]));
    for (const c of data.cases) {
      if (!topics.has(c.topic) || c.status !== "verified") throw new Error(`Unqualified showcase: ${c.id}`);
      if (!["video", "image"].includes(c.kind) || !validMedia(c.media) || !validMedia(c.poster)) throw new Error(`Invalid media: ${c.id}`);
      if (!c.title || !c.doc || !c.category || !c.badge) throw new Error(`Incomplete case: ${c.id}`);
      for (const v of c.views || []) if (!validMedia(v.media) || !v.label) throw new Error(`Invalid view: ${c.id}`);
    }
    for (const t of data.topics) {
      if (!groups.has(t.group) || cases.get(t.heroCase)?.topic !== t.id) throw new Error(`Invalid topic: ${t.id}`);
    }
    for (const g of data.groups) if (g.topics.some((id) => !topics.has(id)) || !cases.has(g.heroCase)) throw new Error(`Invalid group: ${g.id}`);
    return true;
  }
  function getRoute(hash) {
    if (/^#\/(?:topics|team-learning)\/?(?:\?.*)?$/.test(hash)) return { type: "topics", topic: "", anchor: new URLSearchParams(hash.split("?")[1] || "").get("anchor") || "" };
    const m = hash.match(/^#\/(?:topics|team-learning)\/([a-z0-9-]+)(?:\?(.*))?$/);
    if (m) return { type: "topics", topic: m[1], anchor: new URLSearchParams(m[2] || "").get("anchor") || "" };
    if (/^#\/amd-physical-ai(?:\?.*)?$/.test(hash)) return { type: "topics", topic: "amd", anchor: hash.includes("anchor=") ? "topicVideos" : "", legacy: true };
    return null;
  }
  function selectCases(data, topicIds, category = "全部", query = "") {
    const ids = new Set(topicIds);
    const keyword = query.normalize("NFKC").trim().toLocaleLowerCase();
    return data.cases.filter((c) => c.status === "verified" && ids.has(c.topic)
      && (category === "全部" || c.category === category)
      && (!keyword || `${c.title} ${c.description} ${c.badge} ${c.category}`.normalize("NFKC").toLocaleLowerCase().includes(keyword)));
  }
  function collectAssetRefs(data) {
    const refs = new Set();
    for (const c of data.cases) {
      for (const value of [c.media, c.poster, ...(c.views || []).map((v) => v.media)]) {
        if (value && !value.startsWith("https://") && !value.startsWith("topic-media/")) refs.add(value);
      }
    }
    return refs;
  }

  function create({ container, data, fileHref, docHref }) {
    validateData(data);
    const byCase = new Map(data.cases.map((c) => [c.id, c]));
    const byTopic = new Map(data.topics.map((t) => [t.id, t]));
    const byGroup = new Map(data.groups.map((g) => [g.id, g]));
    let events, returnFocus, filter = "全部", query = "", currentTopics = [];
    const mediaUrl = (value) => value.startsWith("https://") || value.startsWith("topic-media/") ? value : fileHref(value);
    const topicLink = (id) => `#/topics/${id}`;
    const docLink = (path) => esc(docHref(path));

    function dispose() {
      container.querySelectorAll("video").forEach((v) => v.pause());
      const dialog = container.querySelector("dialog");
      if (dialog?.open) dialog.close();
      events?.abort();
    }
    function begin() { dispose(); events = new AbortController(); }
    function navigation(active = "") {
      const group = byTopic.get(active)?.group || active;
      return `<nav class="tl-nav" aria-label="专题导航"><div class="tl-wrap tl-nav-inner">
        <a class="tl-nav-root ${!active ? "is-current" : ""}" href="#/topics" ${!active ? 'aria-current="page"' : ""}>专题组队学习</a>
        <span class="tl-nav-divider" aria-hidden="true"></span>
        ${data.groups.map((g) => `<a href="${topicLink(g.id)}" class="${group === g.id ? "is-current" : ""}" ${group === g.id ? 'aria-current="page"' : ""}>${esc(g.title)}</a>`).join("")}
        <a href="#/book?catalog=1" class="tl-nav-book">完整电子书 ${icon("arrow")}</a>
      </div></nav>`;
    }
    function projectCards(topics) {
      return `<div class="tl-project-grid">${topics.map((t) => {
        const hero = byCase.get(t.heroCase), count = selectCases(data, [t.id]).length;
        return `<a class="tl-project-card" href="${topicLink(t.id)}"><div class="tl-project-cover"><img src="${esc(mediaUrl(hero.poster))}" alt="${esc(t.title)} 演示画面" loading="lazy"><span class="tl-project-open">进入专题 ${icon("arrow")}</span></div>
          <div class="tl-project-copy"><span class="tl-eyebrow">${esc(t.eyebrow)}</span><h3>${esc(t.title)}</h3><p>${esc(t.description)}</p><div class="tl-project-meta"><span>${count} 个实践案例</span><span>${esc(t.tags.at(-1))} <b aria-hidden="true">↗</b></span></div></div></a>`;
      }).join("")}</div>`;
    }
    function caseCard(c) {
      const topic = byTopic.get(c.topic);
      return `<article class="tl-case-card"><button class="tl-case-preview" data-tl-case="${esc(c.id)}" aria-label="${c.kind === "video" ? "播放" : "查看"}${esc(c.title)}">
        <img src="${esc(mediaUrl(c.poster))}" alt="${esc(c.title)}" loading="lazy"><span class="tl-case-badge">${esc(c.badge)}</span><span class="tl-play-icon">${icon(c.kind === "video" ? "play" : "arrow")}</span>
        <span class="tl-case-type">${c.kind === "video" ? "完整视频" : c.media.endsWith(".gif") ? "动态效果" : "识别效果图"}${c.views?.length > 1 ? ` · ${c.views.length} 路可切换` : ""}</span></button>
        <div class="tl-case-copy"><div class="tl-case-meta"><span>${esc(topic.title)}</span><span>${esc(c.category)}</span></div><h3><button data-tl-case="${esc(c.id)}">${esc(c.title)}</button></h3><p>${esc(c.description)}</p><a class="tl-text-link" href="${docLink(c.doc)}">动手复现 <span aria-hidden="true">↗</span></a></div></article>`;
    }
    function dialogMarkup() {
      return `<dialog class="tl-dialog" aria-labelledby="tlDialogTitle"><div class="tl-dialog-header"><div><p class="tl-eyebrow" id="tlDialogMeta"></p><h2 id="tlDialogTitle"></h2></div><button class="tl-dialog-close" data-tl-close aria-label="关闭播放器">×</button></div><div id="tlDialogMedia"></div><div class="tl-dialog-footer"><p id="tlDialogNote"></p><div><a class="tl-text-link" id="tlOriginal" target="_blank" rel="noopener noreferrer">打开原片 ↗</a><a class="home-button primary" id="tlDialogDoc">进入复现教程 ↗</a></div></div></dialog>`;
    }
    function footer() {
      return `<footer class="tl-footer tl-wrap"><div><span class="tl-eyebrow">Every-Embodied · 组队学习</span><h2>看过之后，试着亲手做一遍。</h2><p>每个展示都连接到对应教程、代码和可复核的实验记录。</p></div><a class="home-button primary" href="#/book?catalog=1">打开完整电子书 ↗</a></footer>`;
    }
    function renderOverview() {
      begin();
      const featured = ["duck-ladder", "jumper-dance", "aura-household", "amd-PackIdenticalLunches"].map((id) => byCase.get(id));
      container.innerHTML = navigation() + `<div class="tl-wrap"><header class="tl-hub-hero"><div class="tl-hub-copy"><p class="tl-eyebrow"><span class="tl-live-dot"></span> 专题组队学习</p><h1>一起，把机器人<br><em>跑起来。</em></h1><p class="tl-lead">从会走路的小鸭子，到能回应你的 Aura。<br>选一条路线，先看完整演示，再一起动手复现。</p><div class="tl-hero-actions"><a class="home-button primary" href="#/topics?anchor=topicProjects">选择学习专题 <span>↗</span></a><a class="tl-text-link" href="#/book?catalog=1">浏览完整电子书 →</a></div><div class="tl-stats"><span><strong>${data.groups.length}</strong>学习方向</span><span><strong>${data.topics.length}</strong>实践专题</span><span><strong>${data.cases.length}</strong>精选案例</span></div></div>
        <div class="tl-collage" aria-label="专题演示入口">${featured.map((c, i) => `<a class="tl-collage-item tl-collage-${i}" href="${topicLink(c.topic)}"><img src="${esc(mediaUrl(c.poster))}" alt="${esc(c.title)}"><span><small>${esc(byTopic.get(c.topic).eyebrow.split(" · ")[0])}</small><strong>${esc(c.topic === "jumper" ? "Jumper 螃蟹" : byTopic.get(c.topic).title)}</strong>${icon("arrow")}</span></a>`).join("")}</div></header>
        <section class="tl-section" id="topicProjects"><div class="tl-section-head"><div><p class="tl-eyebrow">选择一条学习路线</p><h2>从机器人运动到家庭操作。</h2></div><p>同一本电子书，不同的实践入口。<br>视频、学习计划和教程都在一起。</p></div><div class="tl-directions">${data.groups.map((g, i) => `<a href="${topicLink(g.id)}"><span class="tl-direction-num">0${i + 1}</span><div><strong>${esc(g.title)}</strong><small>${esc(g.subtitle)}</small></div><span>↗</span></a>`).join("")}</div>${projectCards(data.topics)}</section>
        <section class="tl-section tl-vision-section"><div class="tl-section-head"><div><p class="tl-eyebrow">不只会动，也要看得懂</p><h2>让识别结果，也看得见。</h2></div><a class="tl-text-link" href="#/topics/aura?anchor=topicVideos">探索 Aura 的感知案例 ↗</a></div><div class="tl-case-grid">${data.featuredPerception.map((id) => caseCard(byCase.get(id))).join("")}</div></section></div>` + footer() + dialogMarkup();
      bind();
      return "专题组队学习";
    }
    function heroMarkup(topic, hero) {
      const label = topic ? topic.title : "地瓜机器人";
      return `<div class="tl-hero-media"><div class="tl-player-shell"><video id="tlHeroVideo" controls playsinline preload="none" poster="${esc(mediaUrl(hero.poster))}" aria-label="${esc(hero.title)}" src="${esc(mediaUrl(hero.media))}"></video></div>
        <div class="tl-player-caption"><span>${esc(hero.badge)}</span><strong>${esc(hero.title)}</strong><a href="${esc(mediaUrl(hero.media))}" target="_blank" rel="noopener noreferrer" id="tlHeroOriginal" aria-label="打开${esc(label)}完整原片">原片 ↗</a></div>
        ${hero.views?.length > 1 ? `<div class="tl-views" role="group" aria-label="${esc(hero.viewLabel || "视角切换")}"><span>${esc(hero.viewLabel || "视角")}</span>${hero.views.map((v, i) => `<button data-tl-view="${i}" aria-pressed="${i === 0}">${esc(v.label)}</button>`).join("")}</div>` : ""}<p id="tlHeroStatus" class="tl-player-status" role="status">${hero.views?.length > 1 && !hero.viewsSynchronized ? "GPU 与 BPU 是独立推理回合，切换后从头播放。" : ""}</p></div>`;
    }
    function renderTopic(id) {
      const topic = byTopic.get(id), group = byGroup.get(id);
      if (!topic && !group) {
        begin(); container.innerHTML = navigation() + '<div class="tl-wrap tl-empty"><h1>还没有这个专题。</h1><p>可以从已开放的学习路线继续浏览。</p><a class="home-button primary" href="#/topics">返回专题入口</a></div>'; return "专题未找到";
      }
      begin(); filter = "全部"; query = "";
      currentTopics = topic ? [id] : group.topics;
      const hero = byCase.get((topic || group).heroCase);
      const items = selectCases(data, currentTopics);
      const title = topic?.title || group.title;
      const groupId = topic?.group || id;
      const children = byGroup.get(groupId).topics.map((key) => byTopic.get(key));
      container.innerHTML = navigation(id) + `<div class="tl-wrap"><div class="tl-breadcrumb"><a href="#/topics">专题组队学习</a><span>/</span>${topic && topic.group === "rdk" ? `<a href="#/topics/rdk">地瓜机器人</a><span>/</span>` : ""}<span>${esc(title)}</span></div>
        ${groupId === "rdk" ? `<div class="tl-family-tabs" aria-label="地瓜机器人项目"><a href="#/topics/rdk" ${!topic ? 'aria-current="page"' : ""}>全部地瓜机器人</a>${children.map((t) => `<a href="${topicLink(t.id)}" ${id === t.id ? 'aria-current="page"' : ""}>${esc(t.title)}</a>`).join("")}</div>` : ""}
        <header class="tl-topic-hero" style="--tl-accent:${esc(topic?.accent || "#0c9f6d")}"><div class="tl-topic-copy"><p class="tl-eyebrow">${esc(topic?.eyebrow || "MicroDuck × Jumper · 地瓜机器人")}</p><h1>${esc(topic?.lead || "小鸭子与螃蟹，\n各有拿手本领。").replace("\n", "<br>")}</h1><p class="tl-lead">${esc(topic?.description || group.description)}</p><div class="tl-tags">${(topic?.tags || ["双足与六足", "训练与仿真", "端侧部署"]).map((tag) => `<span>${esc(tag)}</span>`).join("")}</div><div class="tl-hero-actions"><button class="home-button primary" data-tl-hero-play>${icon("play")} 播放完整演示</button><a class="tl-text-link" href="${topic ? docLink(topic.doc) : "#/topics/microduck"}">${topic ? "进入学习教程" : "从 MicroDuck 开始"} ↗</a></div><p class="tl-hero-count">${items.length} 个精选案例 <span>·</span> 点击视频，查看完整过程</p></div>${heroMarkup(topic, hero)}</header>
        ${!topic ? `<section class="tl-section"><div class="tl-section-head"><div><p class="tl-eyebrow">两种形态，一起探索</p><h2>选一位机器人伙伴。</h2></div></div>${projectCards(children)}</section>` : ""}
        <section class="tl-section" id="topicVideos"><div class="tl-section-head"><div><p class="tl-eyebrow">完整演示与识别效果</p><h2>看看它都能做什么。</h2></div><span class="tl-result-count" id="tlResultCount">${items.length} 个案例</span></div>
        <div class="tl-filter-bar"><div class="tl-filters" role="group" aria-label="任务分类">${["全部", ...new Set(items.map((c) => c.category))].map((cat) => `<button data-tl-filter="${esc(cat)}" aria-pressed="${cat === filter}">${esc(cat)}</button>`).join("")}</div><label class="tl-search"><span aria-hidden="true">⌕</span><input id="tlSearch" type="search" placeholder="搜索任务或能力" aria-label="搜索本专题案例"></label></div>
        <div class="tl-case-grid" id="tlCaseGrid">${items.map(caseCard).join("")}</div><p class="tl-selection-note">${id === "amd" ? "这里展示选定的完整成功回合；任务协议、正式评测分母与其余结果见复现教程。" : "每个演示保留自己的输入来源与运行方式，具体步骤和验证条件见对应教程。"}</p></section>
        ${topic ? `<section class="tl-section tl-learning"><div class="tl-section-head"><div><p class="tl-eyebrow">从观看到实践</p><h2>跟着这条路线开始。</h2></div><a class="tl-text-link" href="${docLink(topic.plan)}">查看学习安排 ↗</a></div><div class="tl-learning-grid">${topic.learning.map((step, i) => `<a href="${docLink(step.doc)}"><span>0${i + 1}</span><h3>${esc(step.title)}</h3><p>${esc(step.text)}</p><b aria-hidden="true">↗</b></a>`).join("")}</div>${topic.external ? `<a class="tl-external" href="${esc(topic.external)}" target="_blank" rel="noopener noreferrer">继续浏览 AMD 完整成果站 ${icon("arrow")}</a>` : ""}</section>` : ""}
        </div>` + footer() + dialogMarkup();
      bind(hero); return title;
    }
    function updateCases() {
      const items = selectCases(data, currentTopics, filter, query);
      container.querySelector("#tlResultCount").textContent = `${items.length} 个案例`;
      container.querySelector("#tlCaseGrid").innerHTML = items.length ? items.map(caseCard).join("") : '<div class="tl-no-results"><h3>没有找到匹配的案例。</h3><p>换一个关键词，或看看全部演示。</p><button class="home-button" data-tl-reset>显示全部</button></div>';
      container.querySelectorAll("[data-tl-filter]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.tlFilter === filter));
    }
    function switchView(video, row, index, buttonRoot) {
      const wasPlaying = !video.paused, oldTime = row.viewsSynchronized ? video.currentTime : 0;
      const view = row.views[index];
      video.pause(); video.src = mediaUrl(view.media);
      buttonRoot.querySelectorAll("[data-tl-view]").forEach((b) => b.setAttribute("aria-pressed", Number(b.dataset.tlView) === index));
      const original = buttonRoot.querySelector("#tlHeroOriginal, #tlOriginal"); if (original) original.href = mediaUrl(view.media);
      video.addEventListener("loadedmetadata", () => {
        video.currentTime = Math.min(oldTime, Math.max(0, video.duration - 0.1));
        if (wasPlaying) video.play().catch(() => {});
      }, { once: true, signal: events.signal });
      video.load();
    }
    function openCase(id, trigger) {
      const row = byCase.get(id), dialog = container.querySelector("dialog");
      if (!row) return;
      returnFocus = trigger;
      container.querySelectorAll("video").forEach((v) => v.pause());
      container.querySelector("#tlDialogTitle").textContent = row.title;
      container.querySelector("#tlDialogMeta").textContent = `${byTopic.get(row.topic).title} · ${row.badge}`;
      container.querySelector("#tlDialogNote").textContent = row.description + (row.sourceNote ? " " + row.sourceNote : "");
      container.querySelector("#tlOriginal").href = mediaUrl(row.media);
      container.querySelector("#tlOriginal").textContent = row.kind === "video" ? "打开完整原片 ↗" : "打开原图 ↗";
      container.querySelector("#tlDialogDoc").href = docHref(row.doc);
      const host = container.querySelector("#tlDialogMedia");
      host.innerHTML = row.kind === "video" ? `<video controls playsinline preload="metadata" poster="${esc(mediaUrl(row.poster))}" src="${esc(mediaUrl(row.media))}" aria-label="${esc(row.title)}"></video>` : `<img src="${esc(mediaUrl(row.media))}" alt="${esc(row.title)}">`;
      if (row.views?.length > 1) {
        host.insertAdjacentHTML("beforeend", `<div class="tl-views" role="group" aria-label="${esc(row.viewLabel || "视角切换")}">${row.views.map((v, i) => `<button data-tl-view="${i}" aria-pressed="${i === 0}">${esc(v.label)}</button>`).join("")}</div>`);
        host.querySelectorAll("[data-tl-view]").forEach((b) => b.addEventListener("click", () => switchView(host.querySelector("video"), row, Number(b.dataset.tlView), dialog), { signal: events.signal }));
      }
      dialog.showModal();
      host.querySelector("video")?.play().catch(() => {});
    }
    function bind(hero) {
      container.addEventListener("click", (e) => {
        const show = e.target.closest("[data-tl-case]"); if (show) openCase(show.dataset.tlCase, show);
        if (e.target.closest("[data-tl-close]")) container.querySelector("dialog").close();
        const choice = e.target.closest("[data-tl-filter]"); if (choice) { filter = choice.dataset.tlFilter; updateCases(); }
        if (e.target.closest("[data-tl-reset]")) { filter = "全部"; query = ""; container.querySelector("#tlSearch").value = ""; updateCases(); }
        if (e.target.closest("[data-tl-hero-play]")) container.querySelector("#tlHeroVideo")?.play().catch(() => {});
        const view = e.target.closest(".tl-hero-media [data-tl-view]"); if (view && hero) switchView(container.querySelector("#tlHeroVideo"), hero, Number(view.dataset.tlView), container.querySelector(".tl-hero-media"));
      }, { signal: events.signal });
      container.querySelector("#tlSearch")?.addEventListener("input", (e) => { query = e.target.value; updateCases(); }, { signal: events.signal });
      const dialog = container.querySelector("dialog");
      dialog?.addEventListener("close", () => {
        // A native close event can be queued while another case is opening.
        // Never clear the newly opened player's source or steal its focus.
        if (dialog.open) return;
        dialog.querySelectorAll("video").forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); });
        returnFocus?.focus();
      }, { signal: events.signal });
      dialog?.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); }, { signal: events.signal });
      container.querySelector("#tlHeroVideo")?.addEventListener("error", () => { container.querySelector("#tlHeroStatus").textContent = "视频暂未加载，可通过「原片」入口打开。"; }, { signal: events.signal });
    }
    function renderHomeEntries(target) {
      target.innerHTML = `<div class="home-section-head"><p class="home-kicker">专题组队学习</p><h2>选一个项目，一起把它跑通。</h2><p>每个专题都有可跟随的实践路线。</p></div><div class="tl-home-grid">${data.groups.map((g) => {
        const c = byCase.get(g.heroCase);
        return `<a href="${topicLink(g.id)}" class="tl-home-card"><img src="${esc(mediaUrl(c.poster))}" alt="${esc(g.title)}" loading="lazy"><div><span>${esc(g.subtitle)}</span><h3>${esc(g.title)} ↗</h3></div></a>`;
      }).join("")}</div><a class="tl-text-link tl-home-more" href="#/topics">浏览全部专题与演示 →</a>`;
    }
    return { renderOverview, renderTopic, renderHomeEntries, dispose };
  }
  return { getRoute, validateData, selectCases, collectAssetRefs, create, validMedia };
});
