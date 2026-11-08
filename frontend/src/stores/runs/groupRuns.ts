import type { RuntimeEvent } from '@/types/workspace';
import type { Run, RunKind, RunStatus } from '@/types/runs';
import { GENERATION_SOURCE_LABEL, SETTLE_REASON_LABEL, shortId } from '@/stores/runs/labels';

/** 超过这个时长仍没有终态的运行显示为"未见终态"：事件是 best-effort，可能丢失或超出缓冲。 */
export const STALE_AFTER_MS = 10 * 60 * 1000;

interface Group {
  key: string;
  kind: RunKind;
  events: RuntimeEvent[];
}

function findLastEvent(events: RuntimeEvent[], predicate: (event: RuntimeEvent) => boolean): RuntimeEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (predicate(events[index])) return events[index];
  }
  return undefined;
}

function dataString(event: RuntimeEvent | undefined, field: string): string | null {
  const value = event?.data?.[field];
  return typeof value === 'string' && value ? value : null;
}

function kindOfTrace(traceId: string): RunKind {
  if (traceId.startsWith('task-')) return 'chat';
  if (traceId.startsWith('import-')) return 'import';
  if (traceId.startsWith('mcp-')) return 'mcp';
  return 'system';
}

const SYSTEM_OPENING = new Set(['system.starting', 'system.shutting_down']);
const SYSTEM_CLOSING = new Set(['system.ready', 'system.start_failed', 'system.stopped', 'system.stop_failed']);

/**
 * 按一次运行归组：`memory.task.*` 按 task_id；`trace_id` 不是 system 的事件按 trace；
 * 相邻的 `system.*`（连同其间的 `subsystem.operation.*`）合成一次启动或关闭；其余各自一次。
 * `work.*` 不单独归组，之后按 work id 挂到对应的运行。
 */
function groupPrimaryEvents(events: RuntimeEvent[]): { groups: Map<string, Group>; workEvents: RuntimeEvent[] } {
  const groups = new Map<string, Group>();
  const workEvents: RuntimeEvent[] = [];
  let openSystemKey: string | null = null;

  const push = (key: string, kind: RunKind, event: RuntimeEvent) => {
    const group = groups.get(key);
    if (group) group.events.push(event);
    else groups.set(key, { key, kind, events: [event] });
  };

  for (const event of events) {
    const type = event.event_type;
    if (type === 'event.stream.gap') continue;
    if (type.startsWith('work.')) {
      workEvents.push(event);
    } else if (type.startsWith('memory.task.') && event.task_id) {
      push(`task:${event.task_id}`, 'generation', event);
    } else if (event.trace_id && event.trace_id !== 'system') {
      push(`trace:${event.trace_id}`, kindOfTrace(event.trace_id), event);
    } else if (type.startsWith('system.')) {
      if (SYSTEM_OPENING.has(type) || openSystemKey === null) openSystemKey = `system:${event.event_id}`;
      push(openSystemKey, 'system', event);
      if (SYSTEM_CLOSING.has(type)) openSystemKey = null;
    } else if (type.startsWith('subsystem.operation.') && openSystemKey !== null) {
      push(openSystemKey, 'system', event);
    } else if (type.startsWith('maintenance.task.')) {
      push(`event:${event.event_id}`, 'maintenance', event);
    } else if (type.startsWith('topic.')) {
      push(`event:${event.event_id}`, 'topic', event);
    } else {
      push(`event:${event.event_id}`, 'system', event);
    }
  }
  return { groups, workEvents };
}

/**
 * work 事件挂到它所服务的运行：生成 work 挂到同一 task 的生成运行，交互提交挂到交互标识相同的
 * 导入或对话（对话的交互标识就是进程标识），话题折叠各自合成一次运行；挂不上的自成一次运行。
 */
function attachWorkEvents(groups: Map<string, Group>, workEvents: RuntimeEvent[]): void {
  const interactionOwner = new Map<string, string>();
  for (const group of groups.values()) {
    for (const event of group.events) {
      const interactionId = dataString(event, 'interaction_id');
      if (group.kind === 'import' && interactionId) interactionOwner.set(interactionId, group.key);
      if (group.kind === 'chat' && event.process_id) interactionOwner.set(event.process_id, group.key);
    }
  }

  for (const event of workEvents) {
    const workId = event.task_id ?? '';
    let key: string | undefined;
    let kind: RunKind = 'system';
    if (workId.startsWith('memory_generation:')) {
      const taskKey = `task:${workId.slice('memory_generation:'.length)}`;
      if (groups.has(taskKey)) key = taskKey;
    } else if (workId.startsWith('interaction:')) {
      key = interactionOwner.get(workId.slice('interaction:'.length));
    } else if (workId.startsWith('topic_compaction:')) {
      kind = 'topic';
    }
    key ??= `work:${workId || event.event_id}`;
    const group = groups.get(key);
    if (group) group.events.push(event);
    else groups.set(key, { key, kind, events: [event] });
  }

  for (const group of groups.values()) {
    group.events.sort((a, b) => a.sequence - b.sequence);
  }
}

interface Terminal {
  status: RunStatus;
  event: RuntimeEvent;
}

const TERMINAL_BY_KIND: Record<RunKind, Record<string, RunStatus>> = {
  chat: { 'chat.run.completed': 'completed', 'chat.run.cancelled': 'cancelled', 'chat.run.failed': 'failed' },
  import: {
    'import.turn.accepted': 'completed',
    'import.turn.duplicate': 'duplicate',
    'import.turn.skipped': 'skipped',
    'import.turn.deferred': 'deferred',
  },
  mcp: {},
  generation: {
    'memory.task.completed': 'completed',
    'memory.task.cancelled': 'cancelled',
    'memory.task.failed': 'failed',
  },
  maintenance: { 'maintenance.task.completed': 'completed', 'maintenance.task.failed': 'failed' },
  topic: { 'topic.created': 'completed', 'topic.settled': 'completed', 'topic.deleted': 'completed' },
  system: {
    'system.ready': 'completed',
    'system.start_failed': 'failed',
    'system.stopped': 'completed',
    'system.stop_failed': 'failed',
    'subsystem.operation.completed': 'completed',
    'subsystem.operation.failed': 'failed',
  },
};

const WORK_TERMINAL: Record<string, RunStatus> = {
  'work.succeeded': 'completed',
  'work.failed': 'failed',
  'work.dead_lettered': 'failed',
  'work.cancelled': 'cancelled',
  'work.rejected': 'rejected',
};

function findTerminal(kind: RunKind, events: RuntimeEvent[]): Terminal | null {
  if (kind === 'mcp') {
    const operation = findLastEvent(events, (event) => event.event_type === 'actor.operation');
    if (!operation) return null;
    const status = operation.status === 'rejected' ? 'rejected' : operation.status === 'failed' ? 'failed' : 'completed';
    return { status, event: operation };
  }

  const table = TERMINAL_BY_KIND[kind];
  const terminal = findLastEvent(events, (event) => event.event_type in table);
  if (terminal) return { status: table[terminal.event_type], event: terminal };

  // 只有 work 事件的运行：重试排队或重新开始执行之后又回到进行中。
  const lastWork = findLastEvent(events, (event) => event.event_type.startsWith('work.'));
  if (lastWork && lastWork.event_type in WORK_TERMINAL) {
    return { status: WORK_TERMINAL[lastWork.event_type], event: lastWork };
  }
  return null;
}

function describeTitle(kind: RunKind, events: RuntimeEvent[]): { title: string; subtitle: string } {
  const first = events[0];
  const agentId = events.find((event) => event.agent_id)?.agent_id ?? null;
  switch (kind) {
    case 'chat':
      return { title: `对话 · ${agentId ?? '未知 Agent'}`, subtitle: `进程 ${shortId(first.process_id)}` };
    case 'import': {
      const outcome = events.find((event) => event.event_type.startsWith('import.turn.'));
      const source = dataString(outcome, 'source');
      const turn = dataString(outcome, 'external_turn_id');
      return {
        title: source ? `导入 · ${source}` : '导入',
        subtitle: turn ? `轮次 ${shortId(turn)}` : (first.trace_id ?? ''),
      };
    }
    case 'mcp': {
      const tool = events.find((event) => event.span_name)?.span_name ?? 'MCP';
      return { title: `MCP · ${tool}`, subtitle: agentId ?? '' };
    }
    case 'generation': {
      const task = events.find((event) => event.event_type.startsWith('memory.task.'));
      const source = dataString(task, 'source');
      const canonical = dataString(findLastEvent(events, (event) => dataString(event, 'canonical_alias') !== null), 'canonical_alias');
      return {
        title: `生成 · ${source ? (GENERATION_SOURCE_LABEL[source] ?? source) : '未知来源'}`,
        subtitle: dataString(task, 'pending_alias') ?? (canonical ? `→ ${canonical}` : (dataString(task, 'label') ?? '')),
      };
    }
    case 'maintenance':
      return { title: `维护 · ${dataString(first, 'name') ?? first.source ?? '未知任务'}`, subtitle: first.source ?? '' };
    case 'topic': {
      const topic = `话题 ${shortId(first.topic_id)}`;
      if (first.event_type === 'topic.settled') {
        const reason = dataString(first, 'reason');
        return { title: `话题结算 · ${reason ? (SETTLE_REASON_LABEL[reason] ?? reason) : '未知原因'}`, subtitle: topic };
      }
      if (first.event_type === 'topic.deleted') return { title: '话题删除', subtitle: topic };
      if (first.event_type === 'topic.created') return { title: '话题建立', subtitle: topic };
      return { title: '话题折叠', subtitle: first.task_id ?? '' };
    }
    case 'system': {
      if (first.event_type === 'system.starting' || first.event_type === 'system.ready') {
        return { title: '系统启动', subtitle: `${stepCount(events)} 个启动步骤` };
      }
      if (first.event_type.startsWith('system.')) {
        return { title: '系统关闭', subtitle: `${stepCount(events)} 个关闭步骤` };
      }
      if (first.event_type.startsWith('subsystem.operation.')) {
        return { title: `子系统 · ${dataString(first, 'operation_name') ?? '操作'}`, subtitle: first.source ?? '' };
      }
      if (first.event_type.startsWith('work.')) {
        return { title: `队列任务 · ${dataString(first, 'lane') ?? '未知队列'}`, subtitle: first.task_id ?? '' };
      }
      return { title: first.event_type, subtitle: first.trace_id ?? '' };
    }
  }
}

function stepCount(events: RuntimeEvent[]): number {
  const steps = events.map((event) => event.data?.steps).find(Array.isArray);
  return steps ? steps.length : 0;
}

function collectTopics(events: RuntimeEvent[]): string[] {
  const topics = new Set<string>();
  for (const event of events) {
    if (event.topic_id) topics.add(event.topic_id);
    const routed = dataString(event, 'topic_id');
    if (routed) topics.add(routed);
  }
  return [...topics];
}

function buildRun(group: Group, now: number): Run {
  const { events, kind } = group;
  const first = events[0];
  const startedAt = Date.parse(first.timestamp);
  const endedAt = Date.parse(events[events.length - 1].timestamp);
  const terminal = findTerminal(kind, events);
  const reported = terminal?.event.data?.duration_ms;
  const reportedDuration = typeof reported === 'number' ? reported : null;
  // 只有一条事件的运行没有可量的跨度，耗时只取事件自己报告的值。
  let durationMs: number | null = null;
  if (terminal && events.length > 1) durationMs = Math.max(endedAt - startedAt, reportedDuration ?? 0);
  else if (terminal) durationMs = reportedDuration;

  // 不属于任何运行的单条事件是一次瞬时观测，没有"进行中"可言。
  const standalone = group.key.startsWith('event:');
  let status: RunStatus = terminal?.status ?? (standalone ? 'completed' : 'running');
  if (status === 'running' && now - startedAt > STALE_AFTER_MS) status = 'stale';

  return {
    key: group.key,
    kind,
    ...describeTitle(kind, events),
    status,
    reason: terminal && terminal.status !== 'completed' ? (terminal.event.reason ?? null) : null,
    startedAt,
    endedAt,
    durationMs,
    traceId: first.trace_id && first.trace_id !== 'system' && kind !== 'generation' ? first.trace_id : null,
    taskId: kind === 'generation' ? first.task_id : (events.find((event) => event.task_id)?.task_id ?? null),
    agentId: events.find((event) => event.agent_id)?.agent_id ?? null,
    workspaceId: events.find((event) => event.workspace_id)?.workspace_id ?? null,
    topicIds: collectTopics(events),
    originKey: null,
    generationKeys: [],
    events,
  };
}

/** 生成运行的来源：生成任务事件沿用提交者的 trace，指向提交它的那次运行。 */
function linkGenerations(runs: Map<string, Run>): void {
  for (const run of runs.values()) {
    if (run.kind !== 'generation') continue;
    const submitterTrace = run.events.find(
      (event) => event.event_type.startsWith('memory.task.') && event.trace_id && event.trace_id !== 'system',
    )?.trace_id;
    const origin = submitterTrace ? runs.get(`trace:${submitterTrace}`) : undefined;
    if (!origin) continue;
    run.originKey = origin.key;
    origin.generationKeys.push(run.key);
  }
}

/** 把事件列表派生为运行列表，最新开始的在前。`now` 用于判定"未见终态"。 */
export function groupRuns(events: RuntimeEvent[], now: number): Run[] {
  const { groups, workEvents } = groupPrimaryEvents(events);
  attachWorkEvents(groups, workEvents);

  const runs = new Map<string, Run>();
  for (const group of groups.values()) runs.set(group.key, buildRun(group, now));
  linkGenerations(runs);

  return [...runs.values()].sort((a, b) => b.startedAt - a.startedAt || b.events[0].sequence - a.events[0].sequence);
}

export const ISSUE_STATUSES: ReadonlySet<RunStatus> = new Set(['failed', 'deferred', 'rejected', 'stale']);

export interface RunFilters {
  kind: RunKind | 'all';
  onlyActive: boolean;
  onlyIssues: boolean;
  query: string;
}

/** 列表筛选：种类单选、只看进行中、只看异常，文本按标识与标题的子串匹配。 */
export function filterRuns(runs: Run[], filters: RunFilters): Run[] {
  const query = filters.query.trim().toLowerCase();
  return runs.filter((run) => {
    if (filters.kind !== 'all' && run.kind !== filters.kind) return false;
    if (filters.onlyActive && run.status !== 'running') return false;
    if (filters.onlyIssues && !ISSUE_STATUSES.has(run.status)) return false;
    if (!query) return true;
    return [run.title, run.subtitle, run.traceId, run.taskId, run.agentId, ...run.topicIds].some((value) =>
      value?.toLowerCase().includes(query),
    );
  });
}
