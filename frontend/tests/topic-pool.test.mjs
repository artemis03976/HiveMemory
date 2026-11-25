/** 短期话题池的估算与显示：空闲沉淀的剩余时间按秒级配置与毫秒时间戳换算，只对有未沉淀交互的话题计时。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, formatTokens, minutesUntilAutoSettle;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ formatTokens, minutesUntilAutoSettle } = await vite.ssrLoadModule('/src/utils/topicPool.ts'));
});
after(async () => { await vite?.close(); });

test('空闲沉淀的剩余分钟向上取整，到期后为 0', () => {
  const now = Date.UTC(2026, 10, 11, 12, 0);
  const topic = (lastActive) => ({ lastActive, hasUnsettledInteractions: true });
  assert.equal(minutesUntilAutoSettle(topic(now - 3 * 60_000), 3600, now), 57);
  assert.equal(minutesUntilAutoSettle(topic(now - 59 * 60_000 - 30_000), 3600, now), 1);
  assert.equal(minutesUntilAutoSettle(topic(now - 61 * 60_000), 3600, now), 0);
});

test('没有未沉淀交互的话题不计时：沉淀之后话题留在池中，空闲不会再沉淀', () => {
  const now = Date.UTC(2026, 10, 11, 12, 0);
  assert.equal(
    minutesUntilAutoSettle({ lastActive: now - 61 * 60_000, hasUnsettledInteractions: false }, 3600, now),
    null,
  );
});

test('tokens 按量级取不同精度', () => {
  assert.equal(formatTokens(980), '980');
  assert.equal(formatTokens(18_400), '18.4k');
  assert.equal(formatTokens(131_072), '131k');
});
