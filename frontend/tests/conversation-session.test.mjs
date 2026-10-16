/** 会话状态的可观察行为验收；HTTP 边界替换为确定性响应，Zustand 和历史重放保持真实。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, useChatStore, sessionMessages, initial;
const stored = new Map();
const originalFetch = globalThis.fetch;
const originalStorage = globalThis.localStorage;
const originalWindow = globalThis.window;
before(async () => {
  globalThis.localStorage = { getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key) };
  globalThis.window = { localStorage: globalThis.localStorage };
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ useChatStore } = await vite.ssrLoadModule('/src/stores/chat/chatStore.ts'));
  ({ sessionMessages } = await vite.ssrLoadModule('/src/stores/chat/sessionHistory.ts'));
  initial = useChatStore.getState();
});
after(async () => { await vite?.close(); globalThis.fetch = originalFetch; globalThis.localStorage = originalStorage; globalThis.window = originalWindow; });
const reset = (state = {}) => useChatStore.setState({ ...initial, historyReady: true, ...state }, true);
const summary = { session_id: 's1', title: '会话一', turn_count: 1, active_process_id: null };
const turn = { interaction_id: 'p1', agent_id: 'alice', topic_id: 't1', created_at: '2026-10-15T00:00:00Z', payload: {
  user_message: '继续', assistant_final_text: '半截答案', outcome: 'cancelled', turn_events: [
    { kind: 'user_message', sequence: 0, content: '继续' },
    { kind: 'assistant_message', sequence: 1, content: '半截答案' },
    { kind: 'tool_call', sequence: 2, content: 'CALL helper', tool_kind: 'CALL', action_id: 'a1', target: 'helper', tool_args: { task: '帮忙' } },
  ],
} };

test('恢复历史保留顺序、Agent、话题、中断标记和未完成 CALL', () => {
  const messages = sessionMessages({ ...summary, turns: [turn] });
  assert.deepEqual(messages.map(m => m.content), ['继续', '半截答案']);
  assert.equal(messages[1].agent_id, 'alice');
  assert.equal(messages[1].topicId, 't1');
  assert.equal(messages[1].outcome, 'cancelled');
  assert.deepEqual(messages[1].contentBlocks[1], { kind: 'sub_agent', agentId: 'helper', task: '帮忙', status: 'error', contentBlocks: [], finalText: '调用未完成' });
});

test('后端重启后的 404 清空失效选择，并刷新会话列表', async () => {
  reset({ currentSessionId: 'gone' });
  globalThis.fetch = async url => new Response(JSON.stringify(url.endsWith('/gone') ? { message: '不存在' } : []), { status: url.endsWith('/gone') ? 404 : 200 });
  await useChatStore.getState().selectSession('gone');
  assert.equal(useChatStore.getState().currentSessionId, null);
  assert.deepEqual(useChatStore.getState().messages, []);
  assert.match(useChatStore.getState().sessionError, /不存在/);
});

test('历史尚未读完时不能发送，也不能切换到另一会话', async () => {
  reset();
  let release;
  const reads = [];
  globalThis.fetch = async url => { reads.push(url); return await new Promise(resolve => { release = () => resolve(new Response(JSON.stringify({ ...summary, turns: [turn] }))); }); };
  const loading = useChatStore.getState().selectSession('s1');
  await useChatStore.getState().sendMessage('不应发出');
  await useChatStore.getState().selectSession('s2');
  release(); await loading;
  assert.deepEqual(reads, ['/api/v1/sessions/s1']);
  assert.equal(useChatStore.getState().currentSessionId, 's1');
  assert.deepEqual(useChatStore.getState().messages.map(m => m.content), ['继续', '半截答案']);
});

test('无输出停止删除本轮并回填输入，发送携带当前会话与逐轮 Agent', async () => {
  reset({ currentSessionId: 's1', currentAgentId: 'other' });
  let sent;
  globalThis.fetch = async (url, options) => {
    if (url === '/api/v1/sessions') return new Response('[]');
    sent = JSON.parse(options.body);
    return new Response('event: process_id\ndata: {"process_id":"p2","session_id":"s1"}\n\nevent: done\ndata: {"status":"cancelled","stopped":true}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
  };
  await useChatStore.getState().sendMessage('还没开始');
  assert.equal(sent.session_id, 's1');
  assert.equal(sent.agent_id, 'other');
  assert.equal(useChatStore.getState().draft, '还没开始');
  assert.deepEqual(useChatStore.getState().messages, []);
  assert.equal(useChatStore.getState().isStreaming, false);
});

test('有输出的取消保持正文、话题和结局，浏览器不保存消息正文', async () => {
  reset({ currentSessionId: 's1' });
  globalThis.fetch = async url => url === '/api/v1/sessions' ? new Response('[]') : new Response(
    'event: token\ndata: {"content":"半截答案"}\n\nevent: done\ndata: {"status":"cancelled","final_text":"半截答案","topic_id":"t2"}\n\n',
    { headers: { 'Content-Type': 'text/event-stream' } });
  await useChatStore.getState().sendMessage('问题');
  const reply = useChatStore.getState().messages[1];
  assert.equal(reply.content, '半截答案');
  assert.equal(reply.topicId, 't2');
  assert.equal(reply.outcome, 'cancelled');
  assert.deepEqual(JSON.parse(stored.get('chat-store')).state, { currentSessionId: 's1', currentAgentId: initial.currentAgentId });
});

test('恢复 WRITE ACK 是成功，未完成操作不能继续显示执行中', () => {
  const messages = sessionMessages({ ...summary, turns: [{ ...turn, payload: { ...turn.payload,
    turn_events: [
      { kind: 'tool_call', sequence: 0, content: 'WRITE', action_id: 'a', tool_kind: 'WRITE' },
      { kind: 'tool_result', sequence: 1, content: '已接纳', action_id: 'a', status: 'ack' },
      { kind: 'tool_call', sequence: 2, content: 'READ', action_id: 'b', tool_kind: 'READ' },
    ],
  } }] });
  assert.equal(messages[1].contentBlocks[0].action.status, 'success');
  assert.equal(messages[1].contentBlocks[1].action.status, 'error');
});

test('已保存选择但历史读取失败时，不能向不可见历史的会话发送', async () => {
  reset({ currentSessionId: 's1', historyReady: false });
  globalThis.fetch = async () => { throw new Error('不应发送 HTTP'); };
  await useChatStore.getState().sendMessage('不可发送');
  assert.deepEqual(useChatStore.getState().messages, []);
  assert.equal(useChatStore.getState().isStreaming, false);
});

test('停止请求晚于 done 返回错误时，不能覆盖已确认的终态', async () => {
  reset({ currentSessionId: 's1' });
  let output, rejectStop;
  const encoder = new TextEncoder();
  const processStarted = new Promise(resolve => {
    const unsubscribe = useChatStore.subscribe(state => {
      if (state.currentSessionId === 's1' && state.runStatus === 'streaming') {
        unsubscribe(); resolve();
      }
    });
  });
  globalThis.fetch = async url => {
    if (url === '/api/v1/sessions') return new Response('[]');
    if (url === '/api/v1/chat/stop') return new Promise((_, reject) => { rejectStop = reject; });
    return new Response(new ReadableStream({ start(controller) {
      output = controller;
      controller.enqueue(encoder.encode('event: process_id\ndata: {"process_id":"p3","session_id":"s1"}\n\nevent: token\ndata: {"content":"答案"}\n\n'));
    } }), { headers: { 'Content-Type': 'text/event-stream' } });
  };
  const run = useChatStore.getState().sendMessage('问题');
  await processStarted;
  useChatStore.getState().stopStreaming();
  output.enqueue(encoder.encode('event: done\ndata: {"status":"cancelled","final_text":"答案","topic_id":"t3"}\n\n'));
  output.close();
  await run;
  rejectStop(new Error('停止响应丢失'));
  // 让已触发的拒绝链完成，不用固定时长等待业务状态。
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(useChatStore.getState().runStatus, 'cancelled');
  assert.equal(useChatStore.getState().isStreaming, false);
  assert.deepEqual(useChatStore.getState().connection, { status: 'connected', error: null });
});
