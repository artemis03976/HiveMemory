import type { MemoryAtom } from '@/types/memory';

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
