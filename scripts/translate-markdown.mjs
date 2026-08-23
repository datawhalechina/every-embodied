import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;

export function normalizeRepoPath(value) {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

export function isTranslatableSource(sourcePath, config) {
  const normalized = normalizeRepoPath(sourcePath);
  if (!normalized.endsWith(".md") || normalized.startsWith("en/")) return false;
  if (config.excludedPrefixes.some((prefix) => normalized.startsWith(prefix))) return false;
  const topLevel = normalized.split("/", 1)[0];
  return Object.hasOwn(config.chapterMap, topLevel);
}

export function targetPathForSource(sourcePath, config) {
  const normalized = normalizeRepoPath(sourcePath);
  const [topLevel, ...rest] = normalized.split("/");
  const mappedChapter = config.chapterMap[topLevel];
  if (!mappedChapter || rest.length === 0) return null;
  return path.posix.join(config.targetRoot, mappedChapter, ...rest);
}

export function hashText(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function markdownLines(markdown) {
  return markdown.match(/.*(?:\r?\n|$)/g)?.filter(Boolean) ?? [];
}

export function splitMarkdown(markdown, maxBlockChars = 4500) {
  const chunks = [];
  const pending = [];
  let fence = null;
  let frontMatter = false;
  let htmlComment = false;

  const flush = () => {
    if (pending.length > 0) chunks.push({ kind: "translate", text: pending.splice(0).join("") });
  };
  const protect = (line) => {
    flush();
    const previous = chunks.at(-1);
    if (previous?.kind === "protected") previous.text += line;
    else chunks.push({ kind: "protected", text: line });
  };

  const lines = markdownLines(markdown);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (index === 0 && trimmed === "---") {
      frontMatter = true;
      protect(line);
      continue;
    }
    if (frontMatter) {
      protect(line);
      if (trimmed === "---") frontMatter = false;
      continue;
    }

    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);
    if (fence) {
      protect(line);
      if (fenceMatch && fenceMatch[1][0] === fence[0]) fence = null;
      continue;
    }
    if (fenceMatch) {
      fence = fenceMatch[1];
      protect(line);
      continue;
    }

    if (htmlComment) {
      protect(line);
      if (line.includes("-->")) htmlComment = false;
      continue;
    }
    if (line.includes("<!--")) {
      htmlComment = !line.includes("-->");
      protect(line);
      continue;
    }

    if (/^(?: {4}|\t)/.test(line) || /^\s*<(?:img|video|source|iframe|div|table|tr|td|picture)\b/i.test(line)) {
      protect(line);
      continue;
    }
    if (trimmed === "") {
      flush();
      chunks.push({ kind: "protected", text: line });
      continue;
    }
    if (pending.join("").length + line.length > maxBlockChars) flush();
    pending.push(line);
  }
  flush();
  return chunks;
}

export function protectInlineSyntax(text) {
  const values = [];
  const patterns = [
    /(`+)[\s\S]*?\1/g,
    /\]\((?:\\.|[^)\n])+\)/g,
    /\$\$[\s\S]*?\$\$|\$[^$\n]+\$/g,
    /<\/?[A-Za-z][^>\n]*>/g,
    /https?:\/\/[^\s<>)]+/g,
    /\{\{[^}\n]+\}\}|\$\{[^}\n]+\}/g
  ];
  let protectedText = text;
  for (const pattern of patterns) {
    protectedText = protectedText.replace(pattern, (match) => {
      const token = `[[EE_KEEP_${String(values.length).padStart(4, "0")}]]`;
      values.push(match);
      return token;
    });
  }
  return { text: protectedText, values };
}

export function restoreInlineSyntax(text, values) {
  const found = text.match(/\[\[EE_KEEP_\d{4}\]\]/g) ?? [];
  const expected = values.map((_, index) => `[[EE_KEEP_${String(index).padStart(4, "0")}]]`);
  if (found.length !== expected.length || found.some((token, index) => token !== expected[index])) {
    throw new Error("The translation changed protected Markdown tokens");
  }
  return values.reduce(
    (result, value, index) => result.replace(`[[EE_KEEP_${String(index).padStart(4, "0")}]]`, value),
    text
  );
}

function stripModelWrapper(text) {
  const trimmed = text.trim();
  const match = trimmed.match(/^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$/i);
  return match ? match[1] : trimmed;
}

export function buildTranslationPrompt(sourceText, glossary) {
  const terms = glossary
    .split(/\r?\n/)
    .filter((line) => line.trim() && !line.trim().startsWith("#"))
    .map((line) => `- ${line}`)
    .join("\n");
  return [
    "参考下面的固定翻译：",
    terms,
    "",
    "将以下 Markdown 文本翻译为英语。只输出翻译后的结果，不要解释。",
    "严格保持 Markdown 结构、缩进、列表、表格和换行。",
    "所有形如 [[EE_KEEP_0000]] 的占位符必须原样保留，数量和顺序都不能改变。",
    "模型名、仓库名、命令名、参数名和英文缩写保持原样。",
    "",
    sourceText
  ].join("\n");
}

export async function translateMarkdown(markdown, { translate, glossary = "", maxBlockChars = 4500 }) {
  const chunks = splitMarkdown(markdown, maxBlockChars);
  const output = [];
  for (const chunk of chunks) {
    if (chunk.kind === "protected" || !CJK_RE.test(chunk.text)) {
      output.push(chunk.text);
      continue;
    }
    const trailingNewline = /\r?\n$/.test(chunk.text);
    const protectedBlock = protectInlineSyntax(chunk.text);
    const translated = stripModelWrapper(
      await translate(protectedBlock.text, buildTranslationPrompt(protectedBlock.text, glossary))
    );
    let restored = restoreInlineSyntax(translated, protectedBlock.values);
    if (trailingNewline && !restored.endsWith("\n")) restored += "\n";
    output.push(restored);
  }
  return output.join("");
}

function splitDestination(destination) {
  const match = destination.match(/^([^?#]*)([?#][\s\S]*)?$/);
  return { pathname: match?.[1] ?? destination, suffix: match?.[2] ?? "" };
}

function isExternalDestination(destination) {
  return /^(?:[A-Za-z][A-Za-z0-9+.-]*:|#|\/\/|\/)/.test(destination);
}

export function rewriteLocalLinks(markdown, { sourcePath, targetPath, config, translatedTargets = new Set(), root = "." }) {
  const rewrite = (destination) => {
    if (isExternalDestination(destination)) return destination;
    const { pathname, suffix } = splitDestination(destination);
    if (!pathname) return destination;
    const resolvedSource = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), pathname));
    let resolvedTarget = resolvedSource;
    if (resolvedSource.endsWith(".md") && isTranslatableSource(resolvedSource, config)) {
      const candidate = targetPathForSource(resolvedSource, config);
      if (candidate && (translatedTargets.has(candidate) || fs.existsSync(path.join(root, candidate)))) {
        resolvedTarget = candidate;
      }
    }
    const relative = path.posix.relative(path.posix.dirname(targetPath), resolvedTarget) || path.posix.basename(resolvedTarget);
    return `${relative}${suffix}`;
  };

  return markdown
    .replace(/(!?\[[^\]\n]*\]\()([^)\s]+)([^)]*\))/g, (_, prefix, destination, suffix) => `${prefix}${rewrite(destination)}${suffix}`)
    .replace(/((?:src|href)=["'])([^"']+)(["'])/gi, (_, prefix, destination, suffix) => `${prefix}${rewrite(destination)}${suffix}`);
}

export function parseNameStatus(output) {
  const changes = [];
  for (const line of output.split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split("\t");
    const status = fields[0];
    if (status.startsWith("R")) {
      changes.push({ status: "D", path: normalizeRepoPath(fields[1]) });
      changes.push({ status: "A", path: normalizeRepoPath(fields[2]) });
    } else {
      changes.push({ status: status[0], path: normalizeRepoPath(fields[1]) });
    }
  }
  return changes;
}

export function createHttpTranslator(serverUrl) {
  return async (_sourceText, prompt) => {
    const response = await fetch(`${serverUrl.replace(/\/$/, "")}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "Hy-MT2-1.8B",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.2,
        top_p: 0.6,
        top_k: 20,
        repeat_penalty: 1.05,
        max_tokens: 4096,
        stream: false
      })
    });
    if (!response.ok) throw new Error(`Translation server returned ${response.status}: ${await response.text()}`);
    const payload = await response.json();
    const content = payload.choices?.[0]?.message?.content;
    if (!content) throw new Error("Translation server returned an empty response");
    return content;
  };
}

function git(args, root) {
  return execFileSync("git", ["-c", "core.quotepath=false", ...args], {
    cwd: root,
    encoding: "utf8"
  }).trimEnd();
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function parseArgs(argv) {
  const args = { root: process.cwd(), mode: "changed", maxFiles: null, dryRun: false };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--root") args.root = path.resolve(argv[++index]);
    else if (value === "--base") args.base = argv[++index];
    else if (value === "--head") args.head = argv[++index];
    else if (value === "--backfill") args.mode = "backfill";
    else if (value === "--max-files") args.maxFiles = Number(argv[++index]);
    else if (value === "--dry-run") args.dryRun = true;
    else throw new Error(`Unknown argument: ${value}`);
  }
  return args;
}

function listSources(root, config) {
  return git(["ls-files", "*.md"], root)
    .split(/\r?\n/)
    .map(normalizeRepoPath)
    .filter((sourcePath) => isTranslatableSource(sourcePath, config));
}

function selectWork(args, config, state) {
  if (args.mode === "backfill") {
    return listSources(args.root, config)
      .filter((sourcePath) => {
        const content = fs.readFileSync(path.join(args.root, sourcePath), "utf8");
        return state.files[sourcePath]?.sourceHash !== hashText(content);
      })
      .sort((left, right) => left.localeCompare(right, "zh-CN"))
      .map((sourcePath) => ({ status: "M", path: sourcePath }));
  }
  if (!args.base || !args.head) throw new Error("Changed mode requires --base and --head");
  const output = git(["diff", "--name-status", "-M", args.base, args.head, "--", "*.md"], args.root);
  return parseNameStatus(output).filter((change) => isTranslatableSource(change.path, config) || state.files[change.path]);
}

export async function runCli(argv) {
  const args = parseArgs(argv);
  const configPath = path.join(args.root, ".translation", "config.json");
  const statePath = path.join(args.root, ".translation", "state.json");
  const glossaryPath = path.join(args.root, ".translation", "glossary.txt");
  const config = readJson(configPath);
  const state = readJson(statePath);
  const glossary = fs.readFileSync(glossaryPath, "utf8");
  const maxFiles = args.maxFiles || config.maxFilesPerRun;
  const work = selectWork(args, config, state).slice(0, maxFiles);

  console.log(`Selected ${work.length} Markdown file(s) in ${args.mode} mode.`);
  for (const item of work) console.log(`- ${item.status} ${item.path}`);
  if (args.dryRun || work.length === 0) return;

  const translate = createHttpTranslator(process.env.TRANSLATION_SERVER_URL || "http://127.0.0.1:8080");
  const translatedTargets = new Set(Object.values(state.files).map((entry) => entry.target));
  for (const item of work) {
    const existing = state.files[item.path];
    if (item.status === "D") {
      if (existing?.target) fs.rmSync(path.join(args.root, existing.target), { force: true });
      delete state.files[item.path];
      continue;
    }

    const targetPath = targetPathForSource(item.path, config);
    if (!targetPath) continue;
    translatedTargets.add(targetPath);
    const source = fs.readFileSync(path.join(args.root, item.path), "utf8");
    const prepared = rewriteLocalLinks(source, {
      sourcePath: item.path,
      targetPath,
      config,
      translatedTargets,
      root: args.root
    });
    const translated = await translateMarkdown(prepared, {
      translate,
      glossary,
      maxBlockChars: config.maxBlockChars
    });
    const header = [
      "<!-- Generated by the offline translation workflow.",
      `Source: ${item.path}`,
      `Source SHA-256: ${hashText(source)}`,
      `Model: ${config.model}`,
      "Review machine-translated technical claims before relying on them.",
      "-->",
      ""
    ].join("\n");
    const absoluteTarget = path.join(args.root, targetPath);
    fs.mkdirSync(path.dirname(absoluteTarget), { recursive: true });
    fs.writeFileSync(absoluteTarget, `${header}${translated}`, "utf8");
    state.files[item.path] = {
      target: targetPath,
      sourceHash: hashText(source),
      model: config.model,
      modelRevision: config.modelRevision
    };
  }
  fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
  runCli(process.argv.slice(2)).catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}
