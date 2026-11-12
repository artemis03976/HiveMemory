/** 提示的去重：同一条提示在显示期间只出现一次，文字或类型不同的照常追加。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, pushToast;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ pushToast } = await vite.ssrLoadModule('/src/stores/ui/toastStore.ts'));
});
after(async () => { await vite?.close(); });

const shown = [{ id: 'a', message: '保存失败', type: 'error', duration: 6000 }];

test('文字与类型都相同的提示不再追加', () => {
  assert.equal(pushToast(shown, { id: 'b', message: '保存失败', type: 'error' }), null);
});

test('文字或类型不同的提示照常追加', () => {
  assert.equal(pushToast(shown, { id: 'c', message: '保存失败', type: 'warning' }).length, 2);
  assert.equal(pushToast(shown, { id: 'd', message: '已保存', type: 'error' }).length, 2);
});
