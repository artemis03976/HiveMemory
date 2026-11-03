/** 记忆列表客户端：跟随游标取完整个记忆库。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, fetchAllMemories;
const originalFetch = globalThis.fetch;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ fetchAllMemories } = await vite.ssrLoadModule('/src/services/memoryApi.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); });

/** 按游标返回预置页的后端；同一游标第二次出现即失败，避免缺陷表现为死循环。 */
function pagedBackend(pages) {
  const seenCursors = [];
  globalThis.fetch = async (url) => {
    const cursor = new URL(url, 'http://localhost').searchParams.get('cursor');
    if (seenCursors.includes(cursor)) {
      return new Response(JSON.stringify({ detail: 'cursor repeated' }), { status: 500 });
    }
    seenCursors.push(cursor);
    return new Response(JSON.stringify(pages[cursor ?? 'first']), { status: 200 });
  };
  return seenCursors;
}

test('跟随 next_cursor 取完全部页，返回库内总数', async () => {
  const seenCursors = pagedBackend({
    first: { memories: [{ id: 'm1' }, { id: 'm2' }], total: 3, next_cursor: 'c2' },
    c2: { memories: [{ id: 'm3' }], total: 3, next_cursor: null },
  });

  const { memories, total } = await fetchAllMemories();

  assert.deepEqual(memories.map((m) => m.id), ['m1', 'm2', 'm3']);
  assert.equal(total, 3);
  assert.deepEqual(seenCursors, [null, 'c2']);
});
