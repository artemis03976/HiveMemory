/** 运行时事件流的连接状态：后端拒绝如实报错并给出原因，不能显示为事件流未启用。 */
import { test, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, initializeRuntimeEventStream;
const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
const originalEventSource = globalThis.EventSource;
const sources = [];

/** 浏览器 EventSource 的替身：测试直接触发 onopen 与 onerror。 */
class FakeEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  readyState = FakeEventSource.CONNECTING;
  constructor(url) { this.url = url; sources.push(this); }
  addEventListener() {}
  close() { this.readyState = FakeEventSource.CLOSED; }
}

before(async () => {
  globalThis.window = { location: { origin: 'http://localhost' } };
  globalThis.EventSource = FakeEventSource;
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ initializeRuntimeEventStream } = await vite.ssrLoadModule('/src/stores/workspace/runtimeEventStream.ts'));
});
afterEach(() => { globalThis.fetch = originalFetch; });
after(async () => {
  await vite?.close();
  globalThis.window = originalWindow;
  globalThis.EventSource = originalEventSource;
});

/** 最小的 store：状态写入时检查等待条件，不用固定时长等待状态落定。 */
function workspace() {
  let state = {
    runtimeEventConnection: { status: 'disconnected', error: null, connectedAt: null, lastEventId: null },
    _eventSource: null,
  };
  const waiters = [];
  const set = (partial) => {
    state = { ...state, ...partial };
    for (const waiter of [...waiters]) {
      if (waiter.predicate(state)) {
        waiters.splice(waiters.indexOf(waiter), 1);
        waiter.resolve();
      }
    }
  };
  const get = () => state;
  const until = (predicate) => predicate(state)
    ? Promise.resolve()
    : new Promise((resolve) => { waiters.push({ predicate, resolve }); });
  return { set, get, until };
}

const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
const credentialRejected = () => json(
  { error: 'request.management_credential_invalid', detail: '管理接口需要管理凭据：缺少凭据或凭据不正确' }, 401);
const CREDENTIAL_MESSAGE = '管理凭据缺失或已失效，请刷新页面后重试';

test('状态接口被拒绝时报连接错误并给出原因', async () => {
  const { set, get, until } = workspace();
  globalThis.fetch = async () => credentialRejected();
  initializeRuntimeEventStream(set, get);
  await until((state) => state.runtimeEventConnection.status !== 'connecting');
  assert.equal(get().runtimeEventConnection.status, 'error');
  assert.equal(get().runtimeEventConnection.error, CREDENTIAL_MESSAGE);
});

test('浏览器放弃事件流后，按状态接口区分被拒绝、未启用与连接失败', async () => {
  for (const [respond, status, error] of [
    [credentialRejected, 'error', CREDENTIAL_MESSAGE],
    [() => json({ enabled: false }), 'disabled', null],
    [() => json({ enabled: true, buffer_size: 10 }), 'error', '事件流连接失败'],
  ]) {
    const { set, get, until } = workspace();
    globalThis.fetch = async () => json({ enabled: true, buffer_size: 10 });
    initializeRuntimeEventStream(set, get);
    await until((state) => state._eventSource !== null);
    const source = sources.at(-1);
    source.onopen();
    assert.equal(get().runtimeEventConnection.status, 'connected');

    // 例如后端换了管理凭据：浏览器重连时收到非 200 响应，随即关闭事件流。
    globalThis.fetch = async () => respond();
    source.readyState = FakeEventSource.CLOSED;
    source.onerror();
    await until((state) => state.runtimeEventConnection.status !== 'connected');
    assert.equal(get().runtimeEventConnection.status, status);
    if (error) assert.equal(get().runtimeEventConnection.error, error);
  }
});
