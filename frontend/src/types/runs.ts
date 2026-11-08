import type { RuntimeEvent } from '@/types/workspace';

/** 运行页上的运行种类；排队任务挂不上任何运行时归入系统。 */
export type RunKind = 'chat' | 'import' | 'mcp' | 'generation' | 'maintenance' | 'topic' | 'system';

/** 运行的状态取自终态事件；`stale` 是长时间没有终态，不推断为失败。 */
export type RunStatus =
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'deferred'
  | 'rejected'
  | 'duplicate'
  | 'skipped'
  | 'stale';

/** 由运行时事件派生的一次运行；事件按发生顺序排列。 */
export interface Run {
  /** 列表中的稳定键：trace、task 或单条事件的标识。 */
  key: string;
  kind: RunKind;
  title: string;
  subtitle: string;
  status: RunStatus;
  /** 失败、拒绝或推迟时的原因码。 */
  reason: string | null;
  startedAt: number;
  endedAt: number;
  /** 有终态时为总耗时；进行中为 null。 */
  durationMs: number | null;
  traceId: string | null;
  taskId: string | null;
  agentId: string | null;
  workspaceId: string | null;
  topicIds: string[];
  /** 生成运行：提交它的运行的键；其余运行为 null。 */
  originKey: string | null;
  /** 由这次运行提交的生成运行的键。 */
  generationKeys: string[];
  events: RuntimeEvent[];
}

export type EventTone = 'normal' | 'key' | 'ok' | 'warn' | 'error';

/** 时间线上一条事件的中文描述；只由事件字段拼出，不含正文。 */
export interface EventDescription {
  text: string;
  detail: string | null;
  tone: EventTone;
}

export interface GatewayStep {
  stepId: string;
  label: string;
  durationMs: number;
}
