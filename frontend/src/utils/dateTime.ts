import { formatTimeAgo } from './timeAgo';

const WEEK_MS = 7 * 24 * 60 * 60_000;
const pad = (n: number) => String(n).padStart(2, '0');

/** 统一的时间格式：今天只显示时分，今年显示月日与时分，更早的显示完整日期。 */
export function formatDateTime(timestampMs: number, nowMs: number = Date.now()): string {
  const d = new Date(timestampMs);
  const now = new Date(nowMs);
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (d.toDateString() === now.toDateString()) return hm;
  if (d.getFullYear() === now.getFullYear()) return `${d.getMonth() + 1}月${d.getDate()}日 ${hm}`;
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
}

/** 悬停提示里的完整时间。 */
export function formatFullDateTime(timestampMs: number): string {
  const d = new Date(timestampMs);
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 列表中的时间：一周以内用相对时间，更早的用统一格式。 */
export function formatListTime(timestampMs: number, nowMs: number = Date.now()): string {
  return nowMs - timestampMs < WEEK_MS ? formatTimeAgo(timestampMs, nowMs) : formatDateTime(timestampMs, nowMs);
}
