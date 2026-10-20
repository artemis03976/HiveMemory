/** 聊天可选 Agent 列表：内置图纸不在 Profile 接口结果中，由前端固定补入。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, toChatAgents;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ toChatAgents } = await vite.ssrLoadModule('/src/hooks/useAgents.ts'));
});
after(async () => { await vite?.close(); });

const backendAgent = (alias, title) => ({
  id: `uuid-${alias}`, alias, title, summary: '', tags: [], content: '', agent_config: null,
});

test('后端列表没有内置图纸时，omni_doll 补在首位且后端图纸顺序不变', () => {
  const agents = toChatAgents([backendAgent('meal_assistant', '三餐推荐助手'), backendAgent('coder_doll', 'Coder Doll')]);
  assert.deepEqual(agents.map((a) => a.id), ['omni_doll', 'meal_assistant', 'coder_doll']);
});

test('后端已有同名 omni_doll 图纸时不重复补入，以后端标题为准', () => {
  const agents = toChatAgents([backendAgent('coder_doll', 'Coder Doll'), backendAgent('omni_doll', '注入的全能助手')]);
  assert.deepEqual(agents.map((a) => [a.id, a.name]), [['coder_doll', 'Coder Doll'], ['omni_doll', '注入的全能助手']]);
});
