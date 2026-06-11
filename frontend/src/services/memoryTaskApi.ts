import type { MemoryGenerationTask, MemoryTaskListResponse } from '@/types/memoryTask';
import { MOCK_MEMORY_TASKS } from '@/constants/memoryTasks';

function shouldUseMockMemoryTasks(): boolean {
  return import.meta.env.DEV;
}

function getMockMemoryTask(taskId: string): MemoryGenerationTask | null {
  return MOCK_MEMORY_TASKS.find((task) => task.task_id === taskId) ?? null;
}

export async function fetchMemoryTasks(): Promise<MemoryTaskListResponse> {
  try {
    const response = await fetch('/api/v1/memory-tasks');

    if (!response.ok) {
      throw new Error(`Failed to fetch memory tasks: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    if (shouldUseMockMemoryTasks()) {
      console.warn('Memory task API unavailable, using mock data:', error);
      return { tasks: MOCK_MEMORY_TASKS };
    }
    throw error;
  }
}

export async function fetchMemoryTask(taskId: string): Promise<MemoryGenerationTask> {
  try {
    const response = await fetch(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch memory task: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    const mockTask = getMockMemoryTask(taskId);
    if (shouldUseMockMemoryTasks() && mockTask) {
      console.warn(`Memory task API unavailable for ${taskId}, using mock data:`, error);
      return mockTask;
    }
    throw error;
  }
}

export async function cancelMemoryTask(taskId: string): Promise<MemoryGenerationTask> {
  try {
    const response = await fetch(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}/cancel`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel memory task: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    const mockTask = getMockMemoryTask(taskId);
    if (shouldUseMockMemoryTasks() && mockTask) {
      console.warn(`Memory task cancel API unavailable for ${taskId}, using mock cancellation:`, error);
      return {
        ...mockTask,
        cancel_requested: true,
        reason: 'user_requested',
      };
    }
    throw error;
  }
}
