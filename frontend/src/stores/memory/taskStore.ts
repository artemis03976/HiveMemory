import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  cancelMemoryTask as cancelMemoryTaskApi,
  fetchMemoryTask,
  fetchMemoryTasks,
} from '@/services/memoryTaskApi';
import { taskFromEvent } from '@/stores/memory/runtimeEventTaskMapper';
import { mergeTask, sortTasks } from '@/stores/memory/taskReducers';
import type {
  MemoryGenerationTask,
  MemoryGenerationTaskStatus,
  MemoryTaskConnectionState,
  RuntimeEvent,
} from '@/types';

const TERMINAL_STATUSES = new Set<MemoryGenerationTaskStatus>([
  'completed',
  'cancelled',
  'failed',
]);

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
            }, { ...get().tasksById });
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
