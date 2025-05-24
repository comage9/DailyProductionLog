import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: ['bonohouse.p-e.kr'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/access.log': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      }
    }
  },
  define: {
    'process.env': {}
  }
}) 