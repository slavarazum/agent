// @vitest-environment node
import { readdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

// Every chunk a component function loads costs each call to it about 1 ms on
// Convex cloud, so the count per module is pinned. Bundled the way the Convex
// CLI does (convex bundler/debugBundle.js innerEsbuild), with its esbuild,
// which this package doesn't depend on, hence the hand-written type.
const { build } = createRequire(
  createRequire(import.meta.url).resolve("convex/package.json"),
)("esbuild") as {
  build: (options: object) => Promise<{
    metafile: {
      outputs: Record<
        string,
        { entryPoint?: string; imports: { path: string; kind: string }[] }
      >;
    };
  }>;
};
const dir = fileURLToPath(new URL("./", import.meta.url));

// The files the Convex CLI treats as function entry points.
function entryPoints(sub = ""): string[] {
  return readdirSync(path.join(dir, sub), { withFileTypes: true }).flatMap(
    (entry) => {
      const rel = path.posix.join(sub, entry.name);
      if (entry.isDirectory()) {
        return sub === "" && entry.name === "_generated"
          ? []
          : entryPoints(rel);
      }
      const dots = (entry.name.match(/\./g) ?? []).length;
      return entry.name.endsWith(".ts") &&
        dots === 1 &&
        !["schema.ts", "convex.config.ts"].includes(entry.name)
        ? [rel]
        : [];
    },
  );
}

test("each component module loads as many chunks as it did before onSaveMessages", async () => {
  const { metafile } = await build({
    entryPoints: entryPoints().map((entry) => path.join(dir, entry)),
    bundle: true,
    platform: "browser",
    format: "esm",
    target: "esnext",
    outdir: "out",
    outbase: dir,
    conditions: ["convex", "module"],
    write: false,
    splitting: true,
    chunkNames: "_deps/[hash]",
    treeShaking: true,
    minifySyntax: true,
    minifyIdentifiers: true,
    keepNames: true,
    define: { "process.env.NODE_ENV": '"production"' },
    metafile: true,
    logLevel: "silent",
  });
  const chunks = (file: string, seen = new Set<string>()) => {
    for (const { path: imported, kind } of metafile.outputs[file].imports) {
      if (kind !== "import-statement" || seen.has(imported)) continue;
      seen.add(imported);
      chunks(imported, seen);
    }
    return seen.size;
  };
  expect(
    Object.fromEntries(
      Object.entries(metafile.outputs)
        .filter(([, output]) => output.entryPoint)
        .map(([file]) => [path.posix.relative("out", file), chunks(file)])
        .sort(([a], [b]) => (a < b ? -1 : 1)),
    ),
  ).toEqual({
    "apiKeys.js": 2,
    "files.js": 5,
    "messages.js": 9,
    "onSaveMessages.js": 4,
    "streams.js": 7,
    "threads.js": 9,
    "users.js": 9,
    "vector/index.js": 6,
    "vector/tables.js": 2,
  });
}, 30_000);
