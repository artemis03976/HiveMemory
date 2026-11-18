/** 聊天可选 Agent 列表：内置图纸不在 Profile 接口结果中，由前端固定补入；消息中的 Agent 取自这个列表。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, toChatAgents, resolveChatAgent;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ toChatAgents, resolveChatAgent } = await vite.ssrLoadModule('/src/hooks/useAgents.ts'));
});
after(async () => { await vite?.close(); });

const backendAgent = (agent_id, title) => ({
  agent_id, title, summary: '', persona: '', model: 'default', temperature: null, top_p: null,
  enabled: true, updated_at: '2026-01-01T00:00:00Z',
});

test('omni_doll 补在首位且后端图纸顺序不变', () => {
  const agents = toChatAgents([backendAgent('meal_assistant', '三餐推荐助手'), backendAgent('coder_doll', 'Coder Doll')]);
  assert.deepEqual(agents.map((a) => a.id), ['omni_doll', 'meal_assistant', 'coder_doll']);
});

test('消息中的 Agent 取自 Profile 列表，不在列表中时显示 agent_id', () => {
  const agents = toChatAgents([backendAgent('coder_doll', 'Coder Doll')]);
  assert.equal(resolveChatAgent(agents, 'coder_doll').name, 'Coder Doll');
  assert.equal(resolveChatAgent(agents, 'retired_doll').name, 'retired_doll');
});

test('没有起名字的 Profile 以 agent_id 显示', () => {
  const agents = toChatAgents([backendAgent('meal_assistant', '')]);
  assert.equal(resolveChatAgent(agents, 'meal_assistant').name, 'meal_assistant');
});
