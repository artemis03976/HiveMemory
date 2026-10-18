/** 错误解析矩阵归属公共解析器；服务测试只验证状态、取消和握手链路。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, readApiError, FetchSseClient, uploadAttachment, AttachmentApiError;
const originalFetch = globalThis.fetch;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ readApiError } = await vite.ssrLoadModule('/src/services/apiError.ts'));
  ({ FetchSseClient } = await vite.ssrLoadModule('/src/transports/sse/fetchSseClient.ts'));
  ({ uploadAttachment, AttachmentApiError } = await vite.ssrLoadModule('/src/services/attachmentApi.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => { await vite?.close(); });

for (const [name, body, message] of [
  ['字符串 detail', { detail: '附件已失效' }, '附件已失效'],
  ['旧 message', { message: '会话不存在' }, '会话不存在'],
  ['嵌套 detail.message', { detail: { message: '参数错误' } }, '参数错误'],
  ['未知对象', { detail: { token: 'secret' } }, '操作失败 (422)'],
  ['非对象 JSON', ['secret'], '操作失败 (422)'],
  ['空字符串', { detail: '' }, '操作失败 (422)'],
  ['旧校验列表', { detail: [{ loc: ['body', 'password'], type: 'missing', msg: 'secret', input: 'secret', ctx: { secret: true } }] }, 'body.password: 缺少必填字段'],
  ['自定义校验类别', { detail: [{ loc: ['body', 'key'], type: 'secret-validator', msg: 'secret' }] }, 'body.key: 字段取值不合法'],
]) {
  test(`兼容${name}，仅展示约定字段`, async () => {
    assert.deepEqual(await readApiError(new Response(JSON.stringify(body), { status: 422 }), '操作失败'),
      { status: 422, message });
  });
}

test('新结构保留机器码与受控 reason', async () => {
  const response = new Response(JSON.stringify({ error: 'workspace.operation_denied', detail: '当前身份无权执行该操作', reason: 'root_required' }), { status: 403 });
  assert.deepEqual(await readApiError(response), { status: 403, message: '当前身份无权执行该操作', code: 'workspace.operation_denied', reason: 'root_required' });
});

for (const body of ['<html>secret traceback</html>', 'secret traceback']) {
  test('非 JSON 正文回退到固定提示与状态码', async () => {
    assert.deepEqual(await readApiError(new Response(body, { status: 500 }), '操作失败'),
      { status: 500, message: '操作失败 (500)' });
  });
}

test('SSE 握手失败展示安全 detail', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ error: 'workspace.session.busy', detail: '会话正在处理' }), { status: 409 });
  const errors = [], events = [];
  await new FetchSseClient().connect('/chat', {}, { onEvent: event => events.push(event), onError: error => errors.push(error.message) });
  assert.deepEqual(errors, ['会话正在处理']);
  assert.deepEqual(events, []);
});

test('错误正文读取被取消时不触发 SSE onError', async () => {
  let reading, release;
  const started = new Promise(resolve => { reading = resolve; });
  const pending = new Promise((resolve, reject) => { release = reject; });
  globalThis.fetch = async () => ({ ok: false, status: 503, json: () => { reading(); return pending; } });
  const client = new FetchSseClient(), errors = [];
  const connection = client.connect('/chat', {}, { onEvent: () => assert.fail('不应开流'), onError: error => errors.push(error) });
  await started;
  client.disconnect();
  release(new DOMException('aborted', 'AbortError'));
  await connection;
  assert.deepEqual(errors, []);
});

test('上传错误保留状态以维持重试交互', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ error: 'workspace.asset.removed', detail: '附件已移除' }), { status: 410 });
  await assert.rejects(uploadAttachment(new File(['body'], 'notes.txt'), 'op-1'), error => {
    assert.equal(error instanceof AttachmentApiError, true);
    assert.equal(error.status, 410);
    assert.equal(error.message, '附件已移除');
    return true;
  });
});
