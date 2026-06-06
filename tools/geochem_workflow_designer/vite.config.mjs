import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'node:path';

export default defineConfig({
  root: resolve(process.cwd(), 'frontend'),
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        // 后端默认端口已改为 8766，前端代理需要保持一致。
        target: 'http://127.0.0.1:8766',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolve(process.cwd(), 'frontend_dist'),
    emptyOutDir: true,
  },
});
