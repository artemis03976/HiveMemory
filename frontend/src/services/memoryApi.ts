import type { MemoryAtom, MemoryListResponse } from '@/types/memory';

export type UpdateMemoryPatch = Partial<Pick<
  MemoryAtom,
  'title' | 'summary' | 'content' | 'alias' | 'tags'
>>;

export async function fetchMemories(limit = 100): Promise<MemoryListResponse> {
  const response = await fetch(`/api/v1/memories?limit=${encodeURIComponent(limit)}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch memories: ${response.statusText}`);
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to create memory: ${response.statusText}`);
  }
  return response.json();
}

export async function updateMemory(
  id: string,
  patch: UpdateMemoryPatch,
): Promise<MemoryAtom> {
  const response = await fetch(`/api/v1/memories/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });

  if (!response.ok) {
    throw new Error(`Failed to update memory: ${response.statusText}`);
  }

  return response.json();
}

export async function deleteMemory(id: string): Promise<void> {
  const response = await fetch(`/api/v1/memories/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete memory: ${response.statusText}`);
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ positive, source: 'ui.memory_ref' }),
  });

  if (!response.ok) {
    throw new Error(`Failed to record memory feedback: ${response.statusText}`);
  }

  return response.json();
}
