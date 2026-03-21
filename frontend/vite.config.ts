import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173, // Custom port to avoid Windows reserved ranges
    proxy: {
      '/api': {
        target: 'http://localhost:8769', // Custom port to avoid Windows reserved ranges
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying
      },
    },
  },
})
