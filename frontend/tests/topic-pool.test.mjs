/** 短期话题池的估算与显示：话题停下后自动沉淀的剩余时间按秒级配置与毫秒时间戳换算，只对有未沉淀交互的话题计时；详情列出话题的进展并把折叠摘要标为较早内容的摘要，卡片标出进展记录的条数。 */
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

test('自动沉淀的剩余分钟向上取整，到期后为 0', () => {
  const now = Date.UTC(2026, 10, 11, 12, 0);
  const topic = (lastActive) => ({ lastActive, hasUnsettledInteractions: true });
  assert.equal(minutesUntilAutoSettle(topic(now - 3 * 60_000), 3600, now), 57);
  assert.equal(minutesUntilAutoSettle(topic(now - 59 * 60_000 - 30_000), 3600, now), 1);
  assert.equal(minutesUntilAutoSettle(topic(now - 61 * 60_000), 3600, now), 0);
});

test('没有未沉淀交互的话题不计时：沉淀之后话题留在池中，停下再久也不会再沉淀', () => {
  const now = Date.UTC(2026, 10, 11, 12, 0);
  assert.equal(
    minutesUntilAutoSettle({ lastActive: now - 61 * 60_000, hasUnsettledInteractions: false }, 3600, now),
    null,
  );
});

const SETTINGS = { capacity: 5, settleAfterSeconds: 3600, foldTokenThreshold: 131_072, foldRetainBlocks: 2, progressRecordAfterSeconds: 1800 };

function apiTopic(fields) {
  return {
    topic_id: 't1',
    topic_title: '部署脚本',
    last_update: 1_700_000_000,
    has_unsettled_interactions: true,
    progress_record: [],
    unrecorded_turn_count: 0,
    unrecorded_turns: [],
    ...fields,
  };
}

function apiEntry(text, coveredUntil, turnCount, writtenAt = 1_700_000_100) {
  return {
    text, from_turn_id: 'turn-a', to_turn_id: 'turn-b', to_event_seq: 3,
    covered_until: coveredUntil, turn_count: turnCount, written_at: writtenAt,
  };
}

async function renderDetail(raw, settings = SETTINGS) {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { mapTopic } = await vite.ssrLoadModule('/src/services/topicApi.ts');
  const { TopicDetail } = await vite.ssrLoadModule('/src/components/memory/TopicPool.tsx');
  const topic = mapTopic(raw);
  const noop = () => {};
  return renderToStaticMarkup(createElement(TopicDetail, {
    topic, settings, settleMinutes: null, now: topic.lastActive, onReference: noop, onSettle: noop, onDiscard: noop,
  }));
}

test('进展记录的覆盖位置与轮数换算为话题中的轮次区间，写下的时间换算为毫秒', async () => {
  const { mapTopic } = await vite.ssrLoadModule('/src/services/topicApi.ts');

  const topic = mapTopic(apiTopic({
    progress_record: [apiEntry('第一段', 2, 2, 1_700_000_100), apiEntry('第二段', 5, 3, 1_700_000_200)],
    unrecorded_turn_count: 1,
    unrecorded_turns: [{ turn_number: 6, user: '统计一下', assistant: '共 4 次' }],
  }));

  assert.deepEqual(topic.progressRecord, [
    { text: '第一段', fromTurn: 1, toTurn: 2, writtenAt: 1_700_000_100_000 },
    { text: '第二段', fromTurn: 3, toTurn: 5, writtenAt: 1_700_000_200_000 },
  ]);
  assert.equal(topic.unrecordedTurnCount, 1);
  assert.deepEqual(topic.unrecordedTurns, [{ turnNumber: 6, user: '统计一下', assistant: '共 4 次' }]);
});

test('详情按写下的顺序列出进展记录，其后是还没有整理的轮数', async () => {
  const html = await renderDetail(apiTopic({
    progress_record: [apiEntry('讨论了发布顺序', 2, 2), apiEntry('定下先发 staging', 3, 1)],
    unrecorded_turn_count: 2,
  }));

  assert.ok(html.indexOf('讨论了发布顺序') < html.indexOf('定下先发 staging'));
  assert.match(html, /第 1–2 轮/);
  assert.match(html, /第 3 轮<\/span><span>覆盖 1 轮/);
  assert.match(html, /2 条进展记录 · 已整理 3 \/ 5 轮/);
  assert.match(html, /还有 2 轮没有整理/);
});

test('还没有写过一条时，话题的进展是全部还没有整理的轮次，并说明话题停下多久后写第一条', async () => {
  const html = await renderDetail(apiTopic({ unrecorded_turn_count: 14 }));

  assert.match(html, /还没有进展记录 · 14 轮没有整理/);
  assert.match(html, /话题停下 30 分钟后写下第一条/);
  assert.match(html, /还有 14 轮没有整理/);
});

test('全部轮次都已整理时，详情写明已整理，不再显示还没有整理的轮数', async () => {
  const html = await renderDetail(apiTopic({ progress_record: [apiEntry('整理了备份步骤', 4, 4)] }));

  assert.match(html, /1 条进展记录 · 已整理全部 4 轮/);
  assert.match(html, /全部 4 轮都已整理/);
  assert.doesNotMatch(html, /还有 \d+ 轮没有整理/);
});

test('展开还没有整理的轮次时列出节选；后端只交出最近几轮时写明更早的轮数', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { UnrecordedTurnList } = await vite.ssrLoadModule('/src/components/memory/TopicPool.tsx');
  const turns = [
    { turnNumber: 13, user: '删除数据要不要二次确认？', assistant: '要。' },
    { turnNumber: 14, user: '退出时怎么办？', assistant: '先结算再退出。' },
  ];

  const html = renderToStaticMarkup(createElement(UnrecordedTurnList, { turns, total: 14, recordAfter: '话题停下 30 分钟后' }));

  assert.match(html, /只列出最近 2 轮，更早的 12 轮没有展开/);
  assert.ok(html.indexOf('第 13 轮') < html.indexOf('第 14 轮'));
  assert.match(html, /删除数据要不要二次确认？/);
  assert.match(html, /先结算再退出。/);
  assert.match(html, /话题停下 30 分钟后，这些轮会整理成新的一条进展记录/);
});

test('话题列表响应的折叠摘要在详情中标为较早内容的摘要，与话题的进展同时显示', async () => {
  const html = await renderDetail(apiTopic({
    fold_summary: '前几轮改好了打包步骤',
    progress_record: [apiEntry('改好了打包步骤', 3, 3)],
  }));

  assert.match(html, /较早内容的摘要<\/span><p[^>]*>前几轮改好了打包步骤<\/p>/);
  assert.match(html, /改好了打包步骤<\/p>/);
});

test('话题卡片标出进展记录的条数，还没有写过一条时不标', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { mapTopic } = await vite.ssrLoadModule('/src/services/topicApi.ts');
  const { TopicRow } = await vite.ssrLoadModule('/src/components/memory/TopicPool.tsx');
  const row = (raw) => {
    const topic = mapTopic(raw);
    return renderToStaticMarkup(createElement(TopicRow, {
      topic, selected: false, settleMinutes: null, now: topic.lastActive, onSelect: () => {},
    }));
  };

  assert.match(row(apiTopic({ progress_record: [apiEntry('a', 1, 1), apiEntry('b', 3, 2), apiEntry('c', 4, 1)] })), /3 条进展/);
  assert.doesNotMatch(row(apiTopic({ unrecorded_turn_count: 2 })), /条进展/);
});

test('tokens 按量级取不同精度', () => {
  assert.equal(formatTokens(980), '980');
  assert.equal(formatTokens(18_400), '18.4k');
  assert.equal(formatTokens(131_072), '131k');
});
