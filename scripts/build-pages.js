"use strict";

const fs = require("fs/promises");
const fsSync = require("fs");
const path = require("path");
const {
  repoRoot,
  projectAssetMap,
  fullPathFromRel,
  collectCatalog,
  buildDocPayload,
  fetchGithubRepo,
} = require("../web/lib/content");

const distRoot = path.join(repoRoot, "dist");
const siteRoot = path.join(distRoot, "zh-cn");
const publicRoot = path.join(repoRoot, "web", "public");
const katexRoot = path.join(repoRoot, "node_modules", "katex", "dist");

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function emptyDir(dirPath) {
  await fs.rm(dirPath, { recursive: true, force: true });
  await ensureDir(dirPath);
}

async function copyFilePreserveDir(fromPath, toPath) {
  await ensureDir(path.dirname(toPath));
  await fs.copyFile(fromPath, toPath);
}

async function copyDirectory(source, target) {
  await ensureDir(target);
  const entries = await fs.readdir(source, { withFileTypes: true });
  for (const entry of entries) {
    const fromPath = path.join(source, entry.name);
    const toPath = path.join(target, entry.name);
    if (entry.isDirectory()) {
      await copyDirectory(fromPath, toPath);
    } else if (entry.isFile()) {
      await copyFilePreserveDir(fromPath, toPath);
    }
  }
}

function collectFileRefs(value, bucket) {
  const matches = String(value).match(/files\/([^"'()\s?#]+(?:\?[^"'()\s#]*)?)/g) || [];
  for (const match of matches) {
    const relPath = decodeURIComponent(
      match.replace(/^files\//, "").replace(/\?.*$/, ""),
    );
    bucket.add(relPath);
  }
}

function collectHomeAssetRefs() {
  const appSource = fsSync.readFileSync(path.join(publicRoot, "app.js"), "utf8");
  const refs = new Set();
  const matches = appSource.matchAll(/\b(?:media|image|poster):\s*"([^"]+)"/g);
  for (const match of matches) {
    refs.add(match[1]);
  }
  return refs;
}

async function buildSite() {
  await emptyDir(distRoot);
  await ensureDir(siteRoot);

  await copyDirectory(publicRoot, siteRoot);
  await copyDirectory(katexRoot, path.join(siteRoot, "vendor", "katex"));

  const catalog = await collectCatalog({ withVersion: true });
  const github = await fetchGithubRepo().catch(() => ({
    stars: null,
    forks: null,
    updatedAt: null,
  }));

  const dataRoot = path.join(siteRoot, "data");
  const docsRoot = path.join(dataRoot, "docs");
  await ensureDir(docsRoot);

  await fs.writeFile(
    path.join(dataRoot, "catalog.json"),
    JSON.stringify(catalog, null, 2),
    "utf8",
  );
  await fs.writeFile(
    path.join(dataRoot, "github.json"),
    JSON.stringify(github, null, 2),
    "utf8",
  );

  const referencedRepoFiles = collectHomeAssetRefs();
  Object.values(projectAssetMap).forEach((relPath) => {
    if (relPath && fsSync.existsSync(fullPathFromRel(relPath))) {
      referencedRepoFiles.add(relPath);
    }
  });

  for (const doc of catalog.docs) {
    const payload = await buildDocPayload(doc.relPath, catalog, { withVersion: true });
    collectFileRefs(payload.html, referencedRepoFiles);
    await fs.writeFile(
      path.join(docsRoot, `${doc.id}.json`),
      JSON.stringify(payload),
      "utf8",
    );
  }

  for (const relPath of referencedRepoFiles) {
    const sourcePath = fullPathFromRel(relPath);
    if (!fsSync.existsSync(sourcePath)) continue;
    const stat = fsSync.statSync(sourcePath);
    if (!stat.isFile()) continue;
    const toPath = path.join(siteRoot, "files", ...relPath.split("/"));
    await copyFilePreserveDir(sourcePath, toPath);
  }

  const faviconRel = projectAssetMap.favicon;
  if (faviconRel && fsSync.existsSync(fullPathFromRel(faviconRel))) {
    await copyFilePreserveDir(fullPathFromRel(faviconRel), path.join(siteRoot, "favicon.png"));
  }

  const rootRedirect = `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./zh-cn/" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Every-Embodied</title>
  </head>
  <body>
    <p>Redirecting to <a href="./zh-cn/">./zh-cn/</a>...</p>
  </body>
</html>
`;

  const noJekyllPath = path.join(distRoot, ".nojekyll");
  await fs.writeFile(path.join(distRoot, "index.html"), rootRedirect, "utf8");
  await fs.writeFile(noJekyllPath, "", "utf8");

  const siteIndex = await fs.readFile(path.join(siteRoot, "index.html"), "utf8");
  await fs.writeFile(path.join(siteRoot, "404.html"), siteIndex, "utf8");
}

buildSite().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
