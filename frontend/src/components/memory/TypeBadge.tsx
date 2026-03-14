import type { MemoryType } from '@/types/memory';
import { memoryTypeColors, memoryTypeLabels } from '@/types/memory';
import { cn } from '@/lib/utils';

interface TypeBadgeProps {
  type: MemoryType;
  size?: 'sm' | 'md';
}

export function TypeBadge({ type, size = 'md' }: TypeBadgeProps) {
  const { color } = memoryTypeColors[type];
  const label = memoryTypeLabels[type];

  return (
    <span
      className={cn(
        'inline-flex items-center justify-center rounded-full font-medium border',
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
      )}
      style={{
        backgroundColor: `${color}15`,
        borderColor: `${color}40`,
        color: color,
      }}
    >
      {label}
    </span>
  );
}
