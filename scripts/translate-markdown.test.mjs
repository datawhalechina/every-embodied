import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import test from "node:test";
import path from "node:path";
import { execFileSync } from "node:child_process";

import {
  isTranslatableSource,
  parseNameStatus,
  protectInlineSyntax,
  restoreInlineSyntax,
  rewriteLocalLinks,
  splitMarkdown,
  targetPathForSource,
  translateMarkdown
} from "./translate-markdown.mjs";

const config = {
  targetRoot: "en",
  excludedPrefixes: ["20-公众号短文宣发/"],
  chapterMap: {
    "17-具身世界模型": "17-world-models"
  }
};

test("maps configured Chinese chapters without touching the legacy English tree", () => {
  const source = "17-具身世界模型/RoboDream/README.md";
  assert.equal(isTranslatableSource(source, config), true);
  assert.equal(targetPathForSource(source, config), "en/17-world-models/RoboDream/README.md");
  assert.equal(isTranslatableSource("en/ch17/README.md", config), false);
});

test("keeps front matter and fenced code out of translation chunks", () => {
  const markdown = [
    "---\n",
    "title: 中文标题\n",
    "---\n",
    "正文需要翻译。\n",
    "\n",
    "```bash\n",
    "echo 中文不能翻译\n",
    "```\n"
  ].join("");
  const chunks = splitMarkdown(markdown, 1000);
  assert.equal(chunks.filter((chunk) => chunk.kind === "translate").length, 1);
  assert.equal(chunks.find((chunk) => chunk.kind === "translate").text, "正文需要翻译。\n");
  assert.match(chunks.filter((chunk) => chunk.kind === "protected").map((chunk) => chunk.text).join(""), /echo 中文不能翻译/);
});

test("protects and restores inline code, links, formulas, and URLs", () => {
  const source = "运行 `python train.py`，查看 [项目](https://example.com)，计算 $x+y$。";
  const protectedValue = protectInlineSyntax(source);
  assert.doesNotMatch(protectedValue.text, /python train\.py|https:\/\/example\.com|x\+y/);
  assert.equal(restoreInlineSyntax(protectedValue.text, protectedValue.values), source);
  assert.throws(() => restoreInlineSyntax(protectedValue.text.replace("0000", "9999"), protectedValue.values));
});

test("translates prose while preserving protected Markdown exactly", async () => {
  const source = "这是 `VLA` 教程，请访问 [项目](https://example.com)。\n\n```bash\necho 中文\n```\n";
  const translated = await translateMarkdown(source, {
    glossary: "具身智能 = embodied AI",
    maxBlockChars: 1000,
    translate: async (text, prompt) => {
      assert.match(prompt, /只输出翻译后的结果/);
      return text.replace("这是", "This is a").replace("教程，请访问", "tutorial. Visit the").replace("项目", "project");
    }
  });
  assert.match(translated, /This is a `VLA` tutorial/);
  assert.match(translated, /\[project\]\(https:\/\/example\.com\)/);
  assert.match(translated, /```bash\necho 中文\n```/);
});

test("re-bases local assets and prefers an existing translated Markdown target", () => {
  const sourcePath = "17-具身世界模型/topic/README.md";
  const targetPath = "en/17-world-models/topic/README.md";
  const translatedDoc = "en/17-world-models/other.md";
  const markdown = "![图](../assets/demo.png)\n[下一节](../other.md#核心)\n";
  const rewritten = rewriteLocalLinks(markdown, {
    sourcePath,
    targetPath,
    config,
    translatedTargets: new Set([translatedDoc]),
    root: path.parse(process.cwd()).root
  });
  assert.match(rewritten, /\.\.\/\.\.\/\.\.\/17-具身世界模型\/assets\/demo\.png/);
  assert.match(rewritten, /\.\.\/other\.md#核心/);
});

test("expands a rename into deletion and addition", () => {
  assert.deepEqual(parseNameStatus("R100\told.md\tnew.md\nM\tkeep.md"), [
    { status: "D", path: "old.md" },
    { status: "A", path: "new.md" },
    { status: "M", path: "keep.md" }
  ]);
});

test("discovers Chinese paths from git without quoted-path escaping", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "every-embodied-translation-"));
  try {
    fs.mkdirSync(path.join(root, ".translation"));
    fs.mkdirSync(path.join(root, "17-具身世界模型"));
    fs.writeFileSync(path.join(root, ".translation", "config.json"), JSON.stringify(config));
    fs.writeFileSync(path.join(root, ".translation", "state.json"), '{"version":1,"files":{}}');
    fs.writeFileSync(path.join(root, ".translation", "glossary.txt"), "");
    fs.writeFileSync(path.join(root, "17-具身世界模型", "测试.md"), "需要翻译。\n");
    execFileSync("git", ["init", "--quiet"], { cwd: root });
    execFileSync("git", ["add", "."], { cwd: root });

    const messages = [];
    const originalLog = console.log;
    console.log = (message) => messages.push(String(message));
    try {
      const { runCli } = await import("./translate-markdown.mjs");
      await runCli(["--root", root, "--backfill", "--dry-run"]);
    } finally {
      console.log = originalLog;
    }

    assert.ok(messages.includes("Selected 1 Markdown file(s) in backfill mode."));
    assert.ok(messages.some((message) => message.includes("17-具身世界模型/测试.md")));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
