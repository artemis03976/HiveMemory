import { BookOpen, Code2, Link, Lightbulb, Loader, User, Wrench, FileQuestion, type LucideIcon } from 'lucide-react';
import type { MemoryType } from '@/types/memory';

/**
 * 记忆类型的图标块：类型色只上在图标与它的浅底上，类型名是中性文字。
 * 状态总是圆点加文字，两者靠形状区分，所以这里的颜色不会被读成状态。
 */
const TYPE_STYLE: Record<MemoryType, { icon: LucideIcon; tile: string }> = {
  FACT: { icon: BookOpen, tile: 'bg-type-wood/15 text-type-wood' },
  CODE_SNIPPET: { icon: Code2, tile: 'bg-type-water/15 text-type-water' },
  URL_RESOURCE: { icon: Link, tile: 'bg-type-metal/15 text-type-metal' },
  REFLECTION: { icon: Lightbulb, tile: 'bg-type-fire/15 text-type-fire' },
  USER_PROFILE: { icon: User, tile: 'bg-type-moon/15 text-type-moon' },
  WORK_IN_PROGRESS: { icon: Loader, tile: 'bg-type-sun/15 text-type-sun' },
  TOOL: { icon: Wrench, tile: 'bg-type-earth/15 text-type-earth' },
};

const UNKNOWN_STYLE = { icon: FileQuestion, tile: 'bg-ink/10 text-fg-3' };

interface MemoryTypeTileProps {
  type: MemoryType;
  /** sm 用于卡片与列表，lg 用于详情的头部。 */
  size?: 'sm' | 'lg';
}

export default function MemoryTypeTile({ type, size = 'sm' }: MemoryTypeTileProps) {
  const { icon: Icon, tile } = TYPE_STYLE[type] ?? UNKNOWN_STYLE;
  return (
    <span
      aria-hidden="true"
      className={`flex shrink-0 items-center justify-center ${tile} ${size === 'lg' ? 'h-11 w-11 rounded-[13px]' : 'h-8 w-8 rounded-[10px]'}`}
    >
      <Icon className={size === 'lg' ? 'h-[22px] w-[22px]' : 'h-[17px] w-[17px]'} />
    </span>
  );
}
