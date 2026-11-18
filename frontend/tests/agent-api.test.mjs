/** Agent Profile 客户端：以 agent_id 寻址；agent_id 只在新建时提交，修改时出现会被后端拒绝。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, createAgent, saveAgent;
const originalFetch = globalThis.fetch;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ createAgent, saveAgent } = await vite.ssrLoadModule('/src/services/agentApi.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); });

const agent = {
  agentId: 'meal_assistant',
  name: '三餐推荐助手',
  summary: '推荐三餐',
  systemPrompt: '你负责推荐三餐。',
  model: 'deepseek-chat',
  status: 'Inactive',
  config: { model: 'deepseek-chat', temperature: 0.3, top_p: null },
  updatedAt: '2026-01-01T00:00:00Z',
};

/** 记录发出的请求，并以 200 返回一份 Profile。 */
function recordingBackend() {
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ path: new URL(url, 'http://localhost').pathname, method: init.method, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ agent_id: 'meal_assistant' }), { status: 200 });
  };
  return requests;
}

test('保存以 agent_id 寻址，只提交可编辑字段，留空的 top_p 仍为 null', async () => {
  const requests = recordingBackend();

  await saveAgent(agent);

  assert.deepEqual(requests, [{
    path: '/api/v1/agents/meal_assistant',
    method: 'PATCH',
    body: {
      title: '三餐推荐助手', summary: '推荐三餐', persona: '你负责推荐三餐。',
      model: 'deepseek-chat', temperature: 0.3, top_p: null, enabled: false,
    },
  }]);
});

test('新建时提交 agent_id', async () => {
  const requests = recordingBackend();

  await createAgent(agent);

  assert.equal(requests[0].path, '/api/v1/agents');
  assert.equal(requests[0].method, 'POST');
  assert.equal(requests[0].body.agent_id, 'meal_assistant');
});
