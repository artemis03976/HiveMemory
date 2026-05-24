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
