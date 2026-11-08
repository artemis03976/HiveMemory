/** 运行页的派生：事件按一次运行归组、work 挂接、状态取终态，以及时间线描述不含正文。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, groupRuns, filterRuns, describeEvent, STALE_AFTER_MS;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ groupRuns, filterRuns, describeEvent, STALE_AFTER_MS } = await vite.ssrLoadModule('/src/stores/runs/index.ts'));
});
after(async () => { await vite?.close(); });

const BASE = Date.parse('2026-11-07T10:00:00Z');
let sequence = 0;

function ev(eventType, fields = {}, offsetMs = sequence * 10) {
  sequence += 1;
  return {
    event_id: `e${sequence}`,
    sequence,
    event_type: eventType,
    timestamp: new Date(BASE + offsetMs).toISOString(),
    trace_id: 'system',
    span_name: 'main',
    task_type: null,
    source: null,
    subsystem: null,
    component: null,
    severity: 'info',
    process_id: null,
    agent_run_id: null,
    task_id: null,
    agent_id: null,
    frame_id: null,
    topic_id: null,
    atom_id: null,
    workspace_id: null,
    status: null,
    reason: null,
    message: null,
    data: {},
    ...fields,
  };
}

const runByKey = (runs, key) => runs.find((run) => run.key === key);

test('MCP 写入：trace 内的事件成一次运行，它提交的生成任务另成一次运行并互相关联', () => {
  const mcp = { trace_id: 'mcp-1', span_name: 'memory_update', agent_id: 'claude_code' };
  const events = [
    ev('gateway.workflow.started', { ...mcp, data: { ingress_mode: 'passive_memory' } }),
    ev('topic.created', { ...mcp, topic_id: 'topic-a', data: { created_by: 'prepare' } }),
    ev('work.queued', { ...mcp, task_id: 'memory_generation:gen-1' }),
    ev('memory.task.created', { ...mcp, task_id: 'gen-1', topic_id: 'topic-a', data: { source: 'UPDATE', pending_alias: 'rev_x' } }),
    ev('actor.operation', { ...mcp, status: 'completed', data: { operation: 'memory.update', pending_alias: 'rev_x' } }),
    ev('work.succeeded', { task_id: 'memory_generation:gen-1' }),
    ev('memory.task.completed', { task_id: 'gen-1', topic_id: 'topic-a', data: { canonical_alias: 'wip_x' } }),
  ];

  const runs = groupRuns(events, BASE + 1000);
  const mcpRun = runByKey(runs, 'trace:mcp-1');
  const generation = runByKey(runs, 'task:gen-1');

  assert.equal(runs.length, 2);
  assert.equal(mcpRun.title, 'MCP · memory_update');
  assert.equal(mcpRun.status, 'completed');
  assert.deepEqual(mcpRun.events.map((event) => event.event_type), ['gateway.workflow.started', 'topic.created', 'actor.operation']);
  assert.deepEqual(mcpRun.topicIds, ['topic-a']);
  assert.deepEqual(mcpRun.generationKeys, ['task:gen-1']);
  assert.equal(generation.title, '生成 · 更新');
  assert.equal(generation.status, 'completed');
  assert.equal(generation.originKey, 'trace:mcp-1');
  assert.deepEqual(
    generation.events.map((event) => event.event_type),
    ['work.queued', 'memory.task.created', 'work.succeeded', 'memory.task.completed'],
  );
});

test('交互提交的 work 挂到交互标识相同的导入，或进程标识相同的对话', () => {
  const events = [
    ev('import.turn.deferred', { trace_id: 'import-1', reason: 'apply_pending', data: { source: 'claude-code', interaction_id: 'imp-1' } }),
    ev('work.queued', { task_id: 'interaction:imp-1' }),
    ev('chat.run.created', { trace_id: 'task-1', process_id: 'proc-1', agent_id: 'patchouli' }),
    ev('work.queued', { task_id: 'interaction:proc-1' }),
    ev('chat.run.failed', { trace_id: 'task-1', process_id: 'proc-1', reason: 'task_process_failed' }),
  ];

  const runs = groupRuns(events, BASE + 1000);
  const importRun = runByKey(runs, 'trace:import-1');
  const chatRun = runByKey(runs, 'trace:task-1');

  assert.equal(runs.length, 2);
  assert.equal(importRun.title, '导入 · claude-code');
  assert.equal(importRun.status, 'deferred');
  assert.equal(importRun.reason, 'apply_pending');
  assert.ok(importRun.events.some((event) => event.task_id === 'interaction:imp-1'));
  assert.equal(chatRun.title, '对话 · patchouli');
  assert.equal(chatRun.status, 'failed');
  assert.ok(chatRun.events.some((event) => event.task_id === 'interaction:proc-1'));
});

test('挂不上的 work 自成一次运行，状态取 work 的终态', () => {
  const events = [
    ev('work.queued', { task_id: 'interaction:gone', data: { lane: 'patchouli.interaction_submission' } }),
    ev('work.dead_lettered', { task_id: 'interaction:gone', reason: 'apply_failed' }),
  ];

  const [run] = groupRuns(events, BASE + 1000);

  assert.equal(run.key, 'work:interaction:gone');
  assert.equal(run.kind, 'system');
  assert.equal(run.status, 'failed');
});

test('相邻的系统事件合成一次启动或关闭，关闭期间的子系统操作并入关闭', () => {
  const events = [
    ev('system.starting', { data: { steps: ['a', 'b'] } }),
    ev('system.ready', { data: { steps: ['a', 'b'], completed_steps: ['a', 'b'] } }),
    ev('system.shutting_down', { data: { steps: ['c'] } }),
    ev('subsystem.operation.completed', { status: 'completed', data: { operation_name: 'shutdown_drain' } }),
    ev('system.stopped', { data: { steps: ['c'] } }),
  ];

  const runs = groupRuns(events, BASE + 1000);

  assert.deepEqual(runs.map((run) => [run.title, run.status, run.events.length]), [
    ['系统关闭', 'completed', 3],
    ['系统启动', 'completed', 2],
  ]);
});

test('没有终态时为进行中，超过阈值仍没有终态时为未见终态', () => {
  const events = [ev('chat.run.created', { trace_id: 'task-2', process_id: 'proc-2' }, 0)];

  assert.equal(groupRuns(events, BASE + 1000)[0].status, 'running');
  assert.equal(groupRuns(events, BASE + STALE_AFTER_MS + 1)[0].status, 'stale');
});

test('只看异常筛出失败、推迟、拒绝与未见终态的运行', () => {
  const events = [
    ev('actor.operation', { trace_id: 'mcp-ok', status: 'completed', data: { operation: 'memory.read' } }),
    ev('actor.operation', { trace_id: 'mcp-denied', status: 'rejected', reason: 'workspace.operation_denied' }),
    ev('maintenance.task.failed', { reason: 'maintenance_task_failed', data: { name: 'flush' } }),
  ];
  const runs = groupRuns(events, BASE + 1000);

  const issues = filterRuns(runs, { kind: 'all', onlyActive: false, onlyIssues: true, query: '' });

  assert.deepEqual(issues.map((run) => run.status).sort(), ['failed', 'rejected']);
});

test('事件描述只由字段拼出，不带事件 message；未知类型回退为类型名', () => {
  const failed = describeEvent(
    ev('chat.run.failed', { reason: 'task_process_failed', message: '用户消息正文', data: { error_class: 'RuntimeError' } }),
  );
  const read = describeEvent(
    ev('actor.operation', { status: 'completed', data: { operation: 'memory.read', requested_count: 2, delivered_count: 1 } }),
  );
  const unknown = describeEvent(ev('future.event.type'));

  assert.deepEqual(failed, { text: '对话失败：task_process_failed', detail: 'RuntimeError', tone: 'error' });
  assert.equal(read.text, '读取 1/2');
  assert.equal(unknown.text, 'future.event.type');
});
