import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// Vite serves the SPA on :5173 in dev and proxies API calls + MJPEG streams to FastAPI.
// In prod, `pnpm build` outputs to dashboard/dist and OpenPiBot serves the app.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: path.resolve(__dirname, "../dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/camera": { target: "http://127.0.0.1:5000", changeOrigin: true, ws: false },
      "/robot_assets": { target: "http://127.0.0.1:5000", changeOrigin: true },
    },
  },
});
