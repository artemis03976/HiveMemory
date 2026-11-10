/** 接入与权限的接口客户端：冲突原因换成页面上能做的事，访问记录经 root API 携带身份请求头。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, accessApi;
const originalFetch = globalThis.fetch;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  accessApi = await vite.ssrLoadModule('/src/services/accessApi.ts');
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); });

test('登记冲突按原因给出可操作的提示', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({
    error: 'access.registration.conflict',
    detail: '访问登记未修改：与现有登记冲突，或该登记只能经登记文件修改',
    reason: 'file_modified',
  }), { status: 409 });

  await assert.rejects(accessApi.rotatePrincipalCredential('harness:dsh'), (error) => {
    assert.equal(error.status, 409);
    assert.match(error.message, /重启后端/);
    return true;
  });
});

test('用户级记录走自己的路径，访问记录请求携带身份请求头', async () => {
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, method: init.method, headers: init.headers });
    return new Response(JSON.stringify({ agent_id: null, enabled: true, allowed_operations: [] }));
  };

  await accessApi.putActorRecord(null, { enabled: true, allowed_operations: [] });
  await accessApi.putActorRecord('coder doll', { enabled: true, allowed_operations: [] });

  assert.deepEqual(requests.map((r) => r.url), [
    '/api/v1/access/actors/user',
    '/api/v1/access/actors/agents/coder%20doll',
  ]);
  assert.ok(requests.every((r) => r.headers['x-user-id'] === 'default'));
});
