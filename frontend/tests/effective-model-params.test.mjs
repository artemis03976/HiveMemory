/** 当前生效的模型与参数：会话选的模型优先于 Agent，Agent 没给的生成参数取模型定义的默认值。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, effectiveModelParams;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ effectiveModelParams } = await vite.ssrLoadModule('/src/utils/effectiveModelParams.ts'));
});
after(async () => { await vite?.close(); });

const models = [
  { id: 'deepseek-chat', display_name: '', litellm_model: '', provider: 'deepseek', temperature: 1.0, max_tokens: 8192, top_p: 0.95, is_default: true },
  { id: 'qwen-max', display_name: '', litellm_model: '', provider: 'qwen', temperature: 0.7, max_tokens: 4096, top_p: 0.8, is_default: false },
];
const byLabel = (rows) => Object.fromEntries(rows.map((r) => [r.label, [r.value, r.source]]));

test('会话选的模型优先于 Agent 的模型', () => {
  const rows = byLabel(effectiveModelParams('qwen-max', { model_name: 'deepseek-chat' }, models));
  assert.deepEqual(rows['模型'], ['qwen-max', '本次会话']);
  assert.deepEqual(rows['最大输出'], ['4096', '模型默认']);
});

test('Agent 给出的温度优先，没给的参数取模型默认', () => {
  const rows = byLabel(effectiveModelParams(undefined, { model_name: 'deepseek-chat', temperature: 0.3 }, models));
  assert.deepEqual(rows['温度'], ['0.3', '来自 Agent']);
  assert.deepEqual(rows['Top P'], ['0.95', '模型默认']);
});
