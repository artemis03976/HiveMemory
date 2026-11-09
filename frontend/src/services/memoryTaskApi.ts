import { apiErrorMessage } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import type { MemoryGenerationTask, MemoryTaskListResponse } from '@/types/memoryTask';
import { MOCK_MEMORY_TASKS } from '@/constants/memoryTasks';
import { isDebugMode } from '@/utils/debugMode';
import { apiUrl } from '@/transports/backendUrl';

function shouldUseMockMemoryTasks(): boolean {
  return isDebugMode();
}

function getMockMemoryTask(taskId: string): MemoryGenerationTask | null {
  return MOCK_MEMORY_TASKS.find((task) => task.task_id === taskId) ?? null;
}

export async function fetchMemoryTasks(): Promise<MemoryTaskListResponse> {
  try {
    const response = await fetch(apiUrl('/api/v1/memory-tasks'), { headers: identityHeaders() });

    if (!response.ok) {
      throw new Error(await apiErrorMessage(response, 'Failed to fetch memory tasks'));
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
    const response = await fetch(apiUrl(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}`), {
      headers: identityHeaders(),
    });

    if (!response.ok) {
      throw new Error(await apiErrorMessage(response, 'Failed to fetch memory task'));
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
    const response = await fetch(apiUrl(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}/cancel`), {
      method: 'POST',
      headers: identityHeaders(),
    });

    if (!response.ok) {
      throw new Error(await apiErrorMessage(response, 'Failed to cancel memory task'));
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
