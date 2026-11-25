/** 短期话题池的估算与显示：空闲沉淀的剩余时间按秒级配置与毫秒时间戳换算，只对有未沉淀交互的话题计时；详情把折叠摘要标为较早内容的摘要。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, formatTokens, minutesUntilAutoSettle;
const originalStorage = globalThis.localStorage;
const originalWindow = globalThis.window;
before(async () => {
  // 话题池页面导入的持久化 store 在加载时读取 localStorage。
  const stored = new Map();
  globalThis.localStorage = { getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
  globalThis.window = { localStorage: globalThis.localStorage };
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ formatTokens, minutesUntilAutoSettle } = await vite.ssrLoadModule('/src/utils/topicPool.ts'));
});
after(async () => { await vite?.close(); globalThis.localStorage = originalStorage; globalThis.window = originalWindow; });

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

test('话题列表响应的折叠摘要在详情中标为较早内容的摘要', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { mapTopic } = await vite.ssrLoadModule('/src/services/topicApi.ts');
  const { TopicDetail } = await vite.ssrLoadModule('/src/components/memory/TopicPool.tsx');
  const topic = mapTopic({
    topic_id: 't1',
    topic_title: '部署脚本',
    fold_summary: '前几轮改好了打包步骤',
    last_update: 1_700_000_000,
    has_unsettled_interactions: false,
  });
  const noop = () => {};

  const html = renderToStaticMarkup(createElement(TopicDetail, {
    topic, settings: null, settleMinutes: null, now: topic.lastActive, onReference: noop, onSettle: noop, onDiscard: noop,
  }));

  assert.match(html, /较早内容的摘要<\/span><p[^>]*>前几轮改好了打包步骤<\/p>/);
});

test('tokens 按量级取不同精度', () => {
  assert.equal(formatTokens(980), '980');
  assert.equal(formatTokens(18_400), '18.4k');
  assert.equal(formatTokens(131_072), '131k');
});
