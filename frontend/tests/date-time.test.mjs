/** 统一的时间格式：今天只显示时分，今年显示月日与时分，更早的显示完整日期；列表一周内用相对时间。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, formatDateTime, formatListTime;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ formatDateTime, formatListTime } = await vite.ssrLoadModule('/src/utils/dateTime.ts'));
});
after(async () => { await vite?.close(); });

const now = new Date(2026, 10, 11, 15, 30).getTime();

test('按距今远近取不同的精度', () => {
  assert.equal(formatDateTime(new Date(2026, 10, 11, 9, 5).getTime(), now), '09:05');
  assert.equal(formatDateTime(new Date(2026, 9, 2, 14, 20).getTime(), now), '10月2日 14:20');
  assert.equal(formatDateTime(new Date(2025, 0, 16, 4, 30).getTime(), now), '2025年1月16日');
});

test('列表中一周以内用相对时间，更早的用统一格式', () => {
  assert.equal(formatListTime(now - 3 * 60_000, now), '3 分钟前');
  assert.equal(formatListTime(new Date(2026, 9, 2, 14, 20).getTime(), now), '10月2日 14:20');
});
