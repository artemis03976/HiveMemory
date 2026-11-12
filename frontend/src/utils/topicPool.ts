/** 话题的 tokens 显示：千以下原样，十万以下保留一位小数，更大的取整。 */
export function formatTokens(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  if (tokens < 100_000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${Math.round(tokens / 1000)}k`;
}

/**
 * 距离空闲自动结算还剩多少分钟（向上取整，已到期为 0）。按最近一次写入估算：
 * 后端以话题最近一次被确认或写入的时间计时，并由周期扫描结算，实际结算可能稍晚。
 */
export function minutesUntilAutoSettle(lastActiveMs: number, idleTimeoutSeconds: number, nowMs: number): number {
  const remainingMs = lastActiveMs + idleTimeoutSeconds * 1000 - nowMs;
  return remainingMs <= 0 ? 0 : Math.ceil(remainingMs / 60_000);
}
