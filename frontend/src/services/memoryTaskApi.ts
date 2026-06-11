import type { MemoryGenerationTask, MemoryTaskListResponse } from '@/types/memoryTask';

export async function fetchMemoryTasks(): Promise<MemoryTaskListResponse> {
  const response = await fetch('/api/v1/memory-tasks');

  if (!response.ok) {
    throw new Error(`Failed to fetch memory tasks: ${response.statusText}`);
  }

  return response.json();
}

export async function fetchMemoryTask(taskId: string): Promise<MemoryGenerationTask> {
  const response = await fetch(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch memory task: ${response.statusText}`);
  }

  return response.json();
}

export async function cancelMemoryTask(taskId: string): Promise<MemoryGenerationTask> {
  const response = await fetch(`/api/v1/memory-tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel memory task: ${response.statusText}`);
  }

  return response.json();
}
