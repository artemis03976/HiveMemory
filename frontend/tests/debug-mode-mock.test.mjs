/** mock 数据只在 debug 模式下启用：默认关闭时，后端不可用如实报错，不能把假数据当作真实结果。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, fetchMemoryTasks;
const originalFetch = globalThis.fetch;
before(async () => {
  // 开发服务器下 import.meta.env.DEV 为真；debug 模式只看显式的 VITE_DEBUG，这里不设置。
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ fetchMemoryTasks } = await vite.ssrLoadModule('/src/services/memoryTaskApi.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); });

test('未开启 debug 模式时，后端不可用的记忆任务读取如实失败，不返回 mock', async () => {
  globalThis.fetch = async () => { throw new TypeError('fetch failed'); };
  await assert.rejects(fetchMemoryTasks(), /fetch failed/);
});
