/** 记忆可见范围：候选 Agent 的来源合并，以及下拉框取值与读取策略的互换。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, visibilityAgentGroups, policyTarget, targetPolicy;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ visibilityAgentGroups, policyTarget, targetPolicy } = await vite.ssrLoadModule('/src/utils/memoryVisibility.ts'));
});
after(async () => { await vite?.close(); });

test('候选不含用户级记录与保留的 system，同一个 Agent 只出现一次，有 Profile 的归入对话 Agent', () => {
  const groups = visibilityAgentGroups(
    [null, 'system', 'claude_code', 'dsh', 'dsh', 'designer_doll'],
    ['omni_doll', 'designer_doll'],
  );

  assert.deepEqual(groups, { external: ['claude_code', 'dsh'], chat: ['omni_doll', 'designer_doll'] });
});

test('“所有 Agent”对应 PUBLIC，选中某个 Agent 对应只对它可见的 PRIVATE，读回得到同一个取值', () => {
  assert.deepEqual(targetPolicy(''), { visibility: 'PUBLIC' });
  assert.deepEqual(targetPolicy('dsh'), { visibility: 'PRIVATE', target_agent_id: 'dsh' });
  assert.equal(policyTarget(targetPolicy('')), '');
  assert.equal(policyTarget(targetPolicy('dsh')), 'dsh');
});
