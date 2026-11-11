import { ArrowDownToLine, Hash, MessageSquare, Plug, Power, Sparkles, Wrench } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { EventTone, RunKind, RunStatus } from '@/types/runs';

/** 种类只由中性色的图标与名称辨认；颜色只用于状态。 */
export const KIND_META: Record<RunKind, { label: string; icon: LucideIcon }> = {
  chat: { label: '对话', icon: MessageSquare },
  import: { label: '导入', icon: ArrowDownToLine },
  mcp: { label: 'MCP', icon: Plug },
  generation: { label: '生成', icon: Sparkles },
  maintenance: { label: '维护', icon: Wrench },
  topic: { label: '话题', icon: Hash },
  system: { label: '系统', icon: Power },
};

export const KIND_ORDER: RunKind[] = ['chat', 'import', 'mcp', 'generation', 'maintenance', 'topic', 'system'];

/** 状态色沿用七曜状态色在前端已有的对应：水为进行中，木为完成，火为失败，金为等待与警示。 */
export const STATUS_META: Record<RunStatus, { label: string; text: string; dot: string; badge: string }> = {
  running: { label: '进行中', text: 'text-magic-water', dot: 'bg-magic-water animate-pulse', badge: 'bg-magic-water/10' },
  completed: { label: '完成', text: 'text-magic-wood', dot: 'bg-magic-wood', badge: 'bg-magic-wood/10' },
  failed: { label: '失败', text: 'text-magic-fire', dot: 'bg-magic-fire', badge: 'bg-magic-fire/10' },
  deferred: { label: '推迟', text: 'text-magic-metal', dot: 'bg-magic-metal', badge: 'bg-magic-metal/10' },
  rejected: { label: '拒绝', text: 'text-magic-metal', dot: 'bg-magic-metal', badge: 'bg-magic-metal/10' },
  stale: { label: '未见终态', text: 'text-magic-metal', dot: 'bg-magic-metal', badge: 'bg-magic-metal/10' },
  cancelled: { label: '已取消', text: 'text-fg-3', dot: 'bg-fg-3', badge: 'bg-fg-3/10' },
  duplicate: { label: '重复', text: 'text-fg-3', dot: 'bg-fg-3', badge: 'bg-fg-3/10' },
  skipped: { label: '跳过', text: 'text-fg-3', dot: 'bg-fg-3', badge: 'bg-fg-3/10' },
};

/** 时间线上的事件色调：`key` 用 primary 标出运行的起点，其余与状态色一致。 */
export const TONE_META: Record<EventTone, { dot: string; text: string; row: string }> = {
  normal: { dot: 'bg-fg-3', text: 'text-on-surface', row: '' },
  key: { dot: 'bg-primary', text: 'text-on-surface', row: '' },
  ok: { dot: 'bg-magic-wood', text: 'text-on-surface', row: '' },
  warn: { dot: 'bg-magic-metal', text: 'text-magic-metal', row: 'bg-magic-metal/5' },
  error: { dot: 'bg-magic-fire', text: 'text-magic-fire', row: 'bg-magic-fire/5' },
};

function pad(value: number, length = 2): string {
  return value.toString().padStart(length, '0');
}

/** 今天的时间只显示时分秒，更早的带上月日。 */
export function formatClock(epochMs: number, withMillis = false): string {
  const date = new Date(epochMs);
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  const clock = withMillis ? `${time}.${pad(date.getMilliseconds(), 3)}` : time;
  const today = new Date();
  if (date.toDateString() === today.toDateString()) return clock;
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${clock}`;
}

export function formatFullTime(epochMs: number): string {
  return new Date(epochMs).toLocaleString();
}
