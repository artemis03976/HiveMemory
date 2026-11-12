/** 插件卡片的状态推导：已安装优先，其次区分不能代为安装、没有找到 harness 与可以安装。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, pluginCardState, pluginFailureMessage;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ pluginCardState, pluginFailureMessage } = await vite.ssrLoadModule('/src/utils/pluginStatus.ts'));
});
after(async () => { await vite?.close(); });

const status = (patch) => ({
  harness: 'dsh',
  name: 'DeepSeek Harness',
  available: true,
  command: 'dsh.cmd',
  installed: false,
  needs_update: false,
  ...patch,
});

test('已安装的插件即使这里不能代为安装也显示为已安装，版本落后时显示需要更新', () => {
  assert.equal(pluginCardState(status({ installed: true, available: false })), 'installed');
  assert.equal(pluginCardState(status({ installed: true, needs_update: true })), 'needs_update');
});

test('未安装时先看能否代为安装，再看有没有找到 harness', () => {
  assert.equal(pluginCardState(status({ available: false, command: null })), 'unavailable');
  assert.equal(pluginCardState(status({ command: null })), 'missing');
  assert.equal(pluginCardState(status({})), 'not_installed');
});

test('卸载的失败说明接入已经断开，安装的失败说明可以重试或手动安装', () => {
  const failure = { reason: 'command_failed', command: null, output_tail: [] };

  assert.match(pluginFailureMessage(failure, status({}), 'removed'), /已经不能再访问记忆/);
  assert.match(pluginFailureMessage(failure, status({}), 'installed'), /重试/);
});
