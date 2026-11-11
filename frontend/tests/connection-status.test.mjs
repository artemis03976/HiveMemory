/** 连接状况：网络失败置位“连不上后端”、之后的响应清除它；凭据失效一直保留到刷新页面。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, nextConnectionState, reportConnection, localErrorText;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ nextConnectionState, reportConnection, localErrorText } = await vite.ssrLoadModule('/src/transports/connectionStatus.ts'));
});
after(async () => { await vite?.close(); });

const healthy = { unreachable: false, credentialInvalid: false };

test('网络失败后连不上后端，下一次得到响应即恢复', () => {
  const down = nextConnectionState(healthy, 'network-failed');
  assert.equal(down.unreachable, true);
  assert.equal(nextConnectionState(down, 'responded').unreachable, false);
});

test('凭据失效不随之后的响应清除', () => {
  const invalid = nextConnectionState(healthy, 'credential-invalid');
  assert.equal(nextConnectionState(invalid, 'responded').credentialInvalid, true);
});

test('状态条在说明连接问题时，局部读取失败只说数据暂不可用', () => {
  assert.equal(localErrorText('记忆读取失败'), '记忆读取失败');
  reportConnection('network-failed');
  assert.equal(localErrorText('记忆读取失败'), '数据暂不可用');
  reportConnection('responded');
  assert.equal(localErrorText('记忆读取失败'), '记忆读取失败');
  reportConnection('credential-invalid');
  assert.equal(localErrorText('记忆读取失败'), '数据暂不可用');
});
