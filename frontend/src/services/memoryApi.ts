import { apiErrorMessage } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import type { MemoryAtom, MemoryListResponse } from '@/types/memory';

export type UpdateMemoryPatch = Partial<Pick<
  MemoryAtom,
  'title' | 'summary' | 'content' | 'alias' | 'tags'
>>;

export async function fetchMemories(limit = 100): Promise<MemoryListResponse> {
  // Memory 管理请求统一携带 x-user-id / x-workspace-id 基础选择
  const response = await fetch(`/api/v1/memories?limit=${encodeURIComponent(limit)}`, {
    headers: identityHeaders(),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to fetch memories'));
  }

  return response.json();
}

export interface CreateMemoryPayload {
  title: string;
  summary: string;
  content: string;
  memory_type: string;
  tags: string[];
  alias?: string | null;
}

export async function createMemory(payload: CreateMemoryPayload): Promise<MemoryAtom> {
  const response = await fetch('/api/v1/memories', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to create memory'));
  }
  return response.json();
}

export async function updateMemory(
  id: string,
  patch: UpdateMemoryPatch,
): Promise<MemoryAtom> {
  const response = await fetch(`/api/v1/memories/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(patch),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to update memory'));
  }

  return response.json();
}

export async function deleteMemory(id: string): Promise<void> {
  const response = await fetch(`/api/v1/memories/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: identityHeaders(),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to delete memory'));
  }
}

export interface MemoryFeedbackResult {
  success: boolean;
  id: string;
  positive: boolean;
  previous_vitality: number;
  new_vitality: number;
  previous_confidence: number;
  new_confidence: number;
  event_type: string;
}

export async function recordMemoryFeedback(
  memoryId: string,
  positive: boolean,
): Promise<MemoryFeedbackResult> {
  const response = await fetch(`/api/v1/memories/${encodeURIComponent(memoryId)}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify({ positive, source: 'ui.memory_ref' }),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to record memory feedback'));
  }

  return response.json();
}
