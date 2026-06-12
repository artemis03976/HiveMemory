export interface RetryPolicy {
  maxAttempts: number;
  delays: number[];
}

export function getReconnectDelay(attempt: number, policy: RetryPolicy): number | null {
  if (attempt >= policy.maxAttempts) return null;
  return policy.delays[Math.min(attempt, policy.delays.length - 1)] ?? null;
}
