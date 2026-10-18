const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

/** 把毫秒时间戳格式化为相对当前时间的中文描述；未来时间按“刚刚”处理。 */
export function formatTimeAgo(timestampMs: number, nowMs: number = Date.now()): string {
  const elapsed = Math.max(0, nowMs - timestampMs);
  if (elapsed < MINUTE_MS) return '刚刚';
  if (elapsed < HOUR_MS) return `${Math.floor(elapsed / MINUTE_MS)} 分钟前`;
  if (elapsed < DAY_MS) return `${Math.floor(elapsed / HOUR_MS)} 小时前`;
  return `${Math.floor(elapsed / DAY_MS)} 天前`;
}
