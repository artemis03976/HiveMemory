/** 消息列表的跟随判定：停在底部附近时跟随新内容，向上翻过之后不再跟随。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, isNearBottom;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ isNearBottom } = await vite.ssrLoadModule('/src/utils/scrollFollow.ts'));
});
after(async () => { await vite?.close(); });

test('贴底或只差几像素时仍算停在底部', () => {
  assert.equal(isNearBottom(1000, 1600, 600), true);
  assert.equal(isNearBottom(990, 1600, 600), true);
});

test('向上翻过一屏后不再跟随', () => {
  assert.equal(isNearBottom(400, 1600, 600), false);
});
