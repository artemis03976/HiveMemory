import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import fs from 'fs'
import type { ClientRequest } from 'http'
import path from 'path'

// The backend generates this credential on its first start. It is read on every
// forward, so Vite does not need a restart when it starts before the backend or
// after the file is deleted and the backend generates a new one.
const MANAGEMENT_CREDENTIAL_PATH = path.resolve(__dirname, '../configs/management.token')

// Attaches the management credential to a request forwarded to the backend, so the
// browser page never holds it. Without the file nothing is attached and the backend
// answers 401.
function attachManagementCredential(proxyReq: ClientRequest): void {
  let credential: string
  try {
    credential = fs.readFileSync(MANAGEMENT_CREDENTIAL_PATH, 'utf8').trim()
  } catch {
    return
  }
  proxyReq.setHeader('Authorization', `Bearer ${credential}`)
}

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
        configure: (proxy) => {
          proxy.on('proxyReq', attachManagementCredential)
          proxy.on('proxyReqWs', attachManagementCredential)
        },
      },
    },
  },
})
