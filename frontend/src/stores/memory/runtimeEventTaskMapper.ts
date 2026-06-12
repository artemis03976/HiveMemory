import type {
  MemoryGenerationTask,
  MemoryGenerationTaskSource,
  MemoryGenerationTaskStatus,
  RuntimeEvent,
} from '@/types';

export function inferTaskStatus(event: RuntimeEvent): MemoryGenerationTaskStatus | null {
  if (event.status) {
    const status = event.status.toLowerCase();
    if (
      status === 'pending' ||
      status === 'running' ||
      status === 'completed' ||
      status === 'cancelled' ||
      status === 'failed'
    ) {
      return status;
    }
  }

  switch (event.event_type) {
    case 'memory.task.created':
      return 'pending';
    case 'memory.task.status':
      return null;
    case 'memory.task.cancel_requested':
      return null;
    case 'memory.task.cancelled':
      return 'cancelled';
    case 'memory.task.completed':
      return 'completed';
    case 'memory.task.failed':
      return 'failed';
    default:
      return null;
  }
}

export function sourceFromEvent(event: RuntimeEvent): MemoryGenerationTaskSource {
  const source = event.data?.source;
  if (
    source === 'WRITE' ||
    source === 'UPDATE' ||
    source === 'ARCHIVE' ||
    source === 'MERGE' ||
    source === 'SPLIT'
  ) {
    return source;
  }
  return 'WRITE';
}

export function taskFromEvent(event: RuntimeEvent): MemoryGenerationTask | null {
  if (!event.task_id || !event.event_type.startsWith('memory.task.')) {
    return null;
  }

  const now = event.timestamp;
  const status = inferTaskStatus(event) ?? 'pending';
  const data = event.data ?? {};
  const createdAt = typeof data.created_at === 'string' ? data.created_at : now;
  const startedAt = typeof data.started_at === 'string' ? data.started_at : null;
  const finishedAt = typeof data.finished_at === 'string' ? data.finished_at : null;
  const label = typeof data.label === 'string' ? data.label : event.message ?? event.task_id;
  const pendingAlias = typeof data.pending_alias === 'string' ? data.pending_alias : null;
  const canonicalAlias = typeof data.canonical_alias === 'string' ? data.canonical_alias : null;
  const error = typeof data.error === 'string' ? data.error : event.severity === 'error' ? event.message : null;

  return {
    task_id: event.task_id,
    topic_id: event.topic_id ?? '',
    label,
    source: sourceFromEvent(event),
    pending_alias: pendingAlias,
    status,
    canonical_alias: canonicalAlias,
    error,
    created_at: createdAt,
    started_at: startedAt,
    finished_at: finishedAt,
    cancel_requested: event.event_type === 'memory.task.cancel_requested',
    cancelled: status === 'cancelled',
    reason: event.reason,
  };
}
