import type { Topic } from '@/types';

/** 话题的 tokens 显示：千以下原样，十万以下保留一位小数，更大的取整。 */
export function formatTokens(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  if (tokens < 100_000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${Math.round(tokens / 1000)}k`;
}

/**
 * 距离话题停下够久、自动沉淀还剩多少分钟（向上取整，已到期为 0）；话题没有还没沉淀的交互时为 null：
 * 只沉淀新的交互，沉淀之后话题留在池中，停下再久也不再沉淀，直到再次写入。按最近一次
 * 写入估算：后端以话题最近一次被确认或写入的时间计时，并由周期扫描沉淀，实际沉淀可能稍晚。
 */
export function minutesUntilAutoSettle(
  topic: Pick<Topic, 'lastActive' | 'hasUnsettledInteractions'>,
  settleAfterSeconds: number,
  nowMs: number,
): number | null {
  if (!topic.hasUnsettledInteractions) return null;
  const remainingMs = topic.lastActive + settleAfterSeconds * 1000 - nowMs;
  return remainingMs <= 0 ? 0 : Math.ceil(remainingMs / 60_000);
}
