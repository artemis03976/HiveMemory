/** 生成任务事件的映射：来源照实取后端的值，话题结算任务不被显示成写入任务。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, taskFromEvent;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ taskFromEvent } = await vite.ssrLoadModule('/src/stores/memory/runtimeEventTaskMapper.ts'));
});
after(async () => { await vite?.close(); });

test('话题结算产生的生成任务保留 SETTLE 来源', () => {
  const task = taskFromEvent({
    event_type: 'memory.task.created',
    task_id: 'task-1',
    topic_id: 'topic-1',
    timestamp: '2026-11-07T00:00:00Z',
    status: 'pending',
    severity: 'info',
    data: { source: 'SETTLE', label: '结算话题' },
  });

  assert.equal(task.source, 'SETTLE');
});
