import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const packageJson = JSON.parse(
  readFileSync(new URL("./package.json", import.meta.url), "utf-8"),
) as { version?: string };

const appVersion = packageJson.version ?? "dev";

const vendorChunkRules = [
  {
    chunk: "vendor-react",
    paths: [
      "/node_modules/react/",
      "/node_modules/react-dom/",
      "/node_modules/scheduler/",
      "/node_modules/react-router",
      "/node_modules/@remix-run/",
    ],
  },
  {
    chunk: "vendor-radix",
    paths: ["/node_modules/@radix-ui/"],
  },
  {
    chunk: "vendor-motion",
    paths: [
      "/node_modules/framer-motion/",
      "/node_modules/motion-dom/",
      "/node_modules/motion-utils/",
    ],
  },
  {
    chunk: "vendor-markdown",
    paths: [
      "/node_modules/react-markdown/",
      "/node_modules/remark-gfm/",
      "/node_modules/unified/",
      "/node_modules/remark-",
      "/node_modules/rehype-",
      "/node_modules/micromark",
      "/node_modules/mdast-",
      "/node_modules/hast-",
      "/node_modules/unist-",
      "/node_modules/vfile",
      "/node_modules/property-information/",
      "/node_modules/comma-separated-tokens/",
      "/node_modules/space-separated-tokens/",
      "/node_modules/parse-entities/",
      "/node_modules/stringify-entities/",
      "/node_modules/character-entities",
      "/node_modules/decode-named-character-reference/",
    ],
  },
  {
    chunk: "vendor-icons",
    paths: ["/node_modules/lucide-react/"],
  },
  {
    chunk: "vendor-tauri",
    paths: ["/node_modules/@tauri-apps/"],
  },
] as const;

function resolveVendorChunk(id: string): string | undefined {
  const normalizedId = id.replaceAll("\\", "/");
  if (!normalizedId.includes("/node_modules/")) {
    return undefined;
  }

  for (const rule of vendorChunkRules) {
    if (rule.paths.some((path) => normalizedId.includes(path))) {
      return rule.chunk;
    }
  }

  return "vendor";
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  define: {
    __APP_VERSION__: JSON.stringify(appVersion),
  },
  server: {
    port: 3000,
    proxy: {
      "/v1": {
        target: "http://localhost:8080",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks: resolveVendorChunk,
      },
    },
  },
});
