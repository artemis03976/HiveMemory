import type { MemoryGenerationTask } from '@/types';

export function sortTasks(tasks: MemoryGenerationTask[]): MemoryGenerationTask[] {
  return [...tasks].sort((a, b) => {
    const aTime = new Date(a.created_at).getTime();
    const bTime = new Date(b.created_at).getTime();
    return bTime - aTime;
  });
}

export function mergeTask(
  existing: MemoryGenerationTask | undefined,
  incoming: MemoryGenerationTask,
): MemoryGenerationTask {
  return {
    ...existing,
    ...incoming,
    cancel_requested: incoming.cancel_requested ?? existing?.cancel_requested ?? false,
    cancelled: incoming.cancelled ?? incoming.status === 'cancelled',
    reason: incoming.reason ?? existing?.reason ?? null,
  };
}
