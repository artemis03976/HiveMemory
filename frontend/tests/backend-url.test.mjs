/** 后端地址：设置 VITE_BACKEND_ORIGIN 后，HTTP 与 WebSocket 都连到同一个后端，不只迁移日志与事件流。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

const BACKEND = 'https://backend.example:9000';

let vite, fetchMemoryTasks, websocketUrl;
const originalFetch = globalThis.fetch;
before(async () => {
  // Vite 把进程环境中 VITE_ 开头的变量交给 import.meta.env，与 `VITE_BACKEND_ORIGIN=… npm run dev` 一致。
  process.env.VITE_BACKEND_ORIGIN = BACKEND;
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ fetchMemoryTasks } = await vite.ssrLoadModule('/src/services/memoryTaskApi.ts'));
  ({ websocketUrl } = await vite.ssrLoadModule('/src/transports/backendUrl.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); delete process.env.VITE_BACKEND_ORIGIN; });

test('设置后端 origin 时，HTTP 请求与日志 WebSocket 都连到这个 origin', async () => {
  const requested = [];
  globalThis.fetch = async (url) => {
    requested.push(url);
    return new Response(JSON.stringify({ tasks: [] }), { status: 200 });
  };

  await fetchMemoryTasks();

  assert.deepEqual(requested, [`${BACKEND}/api/v1/memory-tasks`]);
  assert.equal(websocketUrl('/api/v1/ws/logs'), 'wss://backend.example:9000/api/v1/ws/logs');
});
