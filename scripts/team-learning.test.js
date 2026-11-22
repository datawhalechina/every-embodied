"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const model = require("../web/public/team-learning");
const data = require("../web/public/team-learning-data.json");

test("all published cases have qualified media and valid topic relationships", () => {
  assert.equal(model.validateData(data), true);
  assert.equal(data.topics.length, 5);
  assert.equal(data.groups.length, 4);
});
test("D-Robotics groups MicroDuck and Jumper, not MagicBox or Aura", () => {
  assert.deepEqual(data.groups.find((g) => g.id === "rdk").topics, ["microduck", "jumper"]);
  const cases = model.selectCases(data, ["microduck", "jumper"]);
  assert(cases.some((c) => c.topic === "jumper"));
  assert(cases.every((c) => c.topic !== "aura"));
});
test("existing AMD URLs keep working, with an equivalent canonical topic", () => {
  assert.equal(model.getRoute("#/amd-physical-ai").topic, "amd");
  assert.equal(model.getRoute("#/amd-physical-ai?anchor=showcaseFrame").anchor, "topicVideos");
  assert.equal(model.getRoute("#/topics/amd").topic, "amd");
});
test("topic deep links and ebook routes remain separate", () => {
  assert.equal(model.getRoute("#/topics").topic, "");
  assert.equal(model.getRoute("#/topics?anchor=topicProjects").anchor, "topicProjects");
  assert.equal(model.getRoute("#/team-learning/microduck").topic, "microduck");
  assert.equal(model.getRoute("#/topics/aura?anchor=topicVideos").anchor, "topicVideos");
  assert.equal(model.getRoute("#/doc/12345"), null);
  assert.equal(model.getRoute("#/book?catalog=1"), null);
});
test("category and text search compose without leaking other topics", () => {
  const selected = model.selectCases(data, ["aura"], "视觉感知", "  YOLO  ");
  assert.deepEqual(selected.map((c) => c.id), ["aura-yolo"]);
  assert.equal(model.selectCases(data, ["jumper"], "全部", "ＹＯＬＯ").length, 0);
  assert.equal(model.selectCases(data, ["aura"], "全部", "ＹＯＬＯ").length, 1);
});
test("inserting a failed run cannot silently reach the showcase", () => {
  const modified = structuredClone(data);
  modified.cases[0].status = "failed";
  assert.throws(() => model.validateData(modified), /Unqualified/);
  assert(!model.selectCases(modified, ["microduck"]).some((c) => c.id === modified.cases[0].id));
});
test("all alternate video views are included by the Pages asset collector", () => {
  const refs = model.collectAssetRefs(data);
  const sample = data.cases.find((c) => c.id === "duck-ladder");
  assert(refs.has(sample.views[1].media));
  assert(refs.has(sample.views[0].media));
  assert([...refs].every((ref) => !ref.startsWith("https:") && !ref.startsWith("topic-media/")));
});
test("media values reject executable schemes, private paths and traversal", () => {
  for (const value of ["javascript:alert(1)", "data:text/html,hi", "file:///tmp/x", "../x.mp4", "C:\\private\\x", "//elsewhere/x"]) assert.equal(model.validMedia(value), false, value);
  assert.equal(model.validMedia("topic-media/aura/perception-full.mp4"), true);
  assert.equal(model.validMedia("https://example.com/full.mp4"), true);
});
test("home recognition entries resolve to real cases", () => {
  for (const id of data.featuredPerception) assert(data.cases.some((c) => c.id === id && c.category === "视觉感知"));
});
test("same-timeline cameras and independent inference runs are distinguished", () => {
  assert.equal(data.cases.find((c) => c.id === "aura-household").viewsSynchronized, true);
  assert.equal(data.cases.find((c) => c.id === "duck-ladder").viewsSynchronized, false);
});
test("new public assets exist without relying on an author-local path", () => {
  for (const c of data.cases) {
    for (const ref of [c.poster, c.media]) if (ref.startsWith("topic-media/")) assert(fs.existsSync(path.join(__dirname, "../web/public", ref)), ref);
    assert(!/127\.0\.0\.1|192\.168\.|\/data\/Data|C:\//.test(JSON.stringify(c)));
  }
});
