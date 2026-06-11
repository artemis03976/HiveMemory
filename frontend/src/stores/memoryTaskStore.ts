import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  cancelMemoryTask as cancelMemoryTaskApi,
  fetchMemoryTask,
  fetchMemoryTasks,
} from '@/services/memoryTaskApi';
import type {
  MemoryGenerationTask,
  MemoryGenerationTaskSource,
  MemoryGenerationTaskStatus,
  MemoryTaskConnectionState,
  RuntimeEvent,
} from '@/types';

const TERMINAL_STATUSES = new Set<MemoryGenerationTaskStatus>([
  'completed',
  'cancelled',
  'failed',
]);

function sortTasks(tasks: MemoryGenerationTask[]): MemoryGenerationTask[] {
  return [...tasks].sort((a, b) => {
    const aTime = new Date(a.created_at).getTime();
    const bTime = new Date(b.created_at).getTime();
    return bTime - aTime;
  });
}

function mergeTask(
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

function inferTaskStatus(event: RuntimeEvent): MemoryGenerationTaskStatus | null {
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

function sourceFromEvent(event: RuntimeEvent): MemoryGenerationTaskSource {
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

function taskFromEvent(event: RuntimeEvent): MemoryGenerationTask | null {
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

interface MemoryTaskStore {
  tasks: MemoryGenerationTask[];
  tasksById: Record<string, MemoryGenerationTask>;
  connection: MemoryTaskConnectionState;
  selectedTaskId: string | null;
  showTerminalTasks: boolean;

  fetchTasks: () => Promise<void>;
  refreshTask: (taskId: string) => Promise<void>;
  refreshTasksByIds: (taskIds: string[]) => Promise<void>;
  cancelTask: (taskId: string) => Promise<void>;
  upsertTask: (task: MemoryGenerationTask) => void;
  applyRuntimeEvent: (event: RuntimeEvent) => void;
  setSelectedTaskId: (taskId: string | null) => void;
  setShowTerminalTasks: (show: boolean) => void;
  clearTasks: () => void;
}

export const useMemoryTaskStore = create<MemoryTaskStore>()(
  devtools(
    persist(
      (set, get) => ({
        tasks: [],
        tasksById: {},
        connection: {
          status: 'idle',
          error: null,
          lastFetchedAt: null,
        },
        selectedTaskId: null,
        showTerminalTasks: true,

        fetchTasks: async () => {
          set((state) => ({
            connection: {
              ...state.connection,
              status: 'loading',
              error: null,
            },
          }));

          try {
            const { tasks } = await fetchMemoryTasks();
            const tasksById = tasks.reduce<Record<string, MemoryGenerationTask>>((acc, task) => {
              acc[task.task_id] = mergeTask(get().tasksById[task.task_id], task);
              return acc;
            }, {});
            set({
              tasks: sortTasks(Object.values(tasksById)),
              tasksById,
              connection: {
                status: 'ready',
                error: null,
                lastFetchedAt: Date.now(),
              },
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch memory tasks';
            set((state) => ({
              connection: {
                ...state.connection,
                status: 'error',
                error: message,
              },
            }));
          }
        },

        refreshTask: async (taskId) => {
          try {
            const task = await fetchMemoryTask(taskId);
            get().upsertTask(task);
            set((state) => ({
              connection: {
                ...state.connection,
                status: 'ready',
                error: null,
                lastFetchedAt: Date.now(),
              },
            }));
          } catch (error) {
            const message = error instanceof Error ? error.message : `Failed to refresh memory task ${taskId}`;
            set((state) => ({
              connection: {
                ...state.connection,
                status: 'error',
                error: message,
              },
            }));
          }
        },

        refreshTasksByIds: async (taskIds) => {
          const uniqueIds = [...new Set(taskIds.filter(Boolean))];
          await Promise.all(uniqueIds.map((taskId) => get().refreshTask(taskId)));
        },

        cancelTask: async (taskId) => {
          const existing = get().tasksById[taskId];
          if (existing && TERMINAL_STATUSES.has(existing.status)) return;

          if (existing) {
            get().upsertTask({
              ...existing,
              cancel_requested: true,
              reason: 'user_requested',
            });
          }

          try {
            const task = await cancelMemoryTaskApi(taskId);
            get().upsertTask(task);
          } catch (error) {
            const message = error instanceof Error ? error.message : `Failed to cancel memory task ${taskId}`;
            if (existing) {
              get().upsertTask({
                ...existing,
                cancel_requested: false,
                error: message,
              });
            }
            set((state) => ({
              connection: {
                ...state.connection,
                status: 'error',
                error: message,
              },
            }));
          }
        },

        upsertTask: (task) => {
          set((state) => {
            const existing = state.tasksById[task.task_id];
            const merged = mergeTask(existing, task);
            const tasksById = {
              ...state.tasksById,
              [task.task_id]: merged,
            };
            return {
              tasksById,
              tasks: sortTasks(Object.values(tasksById)),
            };
          });
        },

        applyRuntimeEvent: (event) => {
          const task = taskFromEvent(event);
          if (!task) return;

          set((state) => {
            const existing = state.tasksById[task.task_id];
            const merged = mergeTask(existing, {
              ...task,
              cancel_requested:
                event.event_type === 'memory.task.cancel_requested'
                  ? true
                  : task.cancel_requested ?? existing?.cancel_requested,
            });
            const tasksById = {
              ...state.tasksById,
              [task.task_id]: merged,
            };
            return {
              tasksById,
              tasks: sortTasks(Object.values(tasksById)),
            };
          });
        },

        setSelectedTaskId: (taskId) => set({ selectedTaskId: taskId }),

        setShowTerminalTasks: (show) => set({ showTerminalTasks: show }),

        clearTasks: () => set({
          tasks: [],
          tasksById: {},
          selectedTaskId: null,
        }),
      }),
      {
        name: 'memory-task-store',
        version: 1,
        partialize: (state) => ({
          showTerminalTasks: state.showTerminalTasks,
        }),
      },
    ),
    { name: 'MemoryTaskStore' },
  ),
);
