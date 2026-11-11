import { Flame, Target, Pin, Edit2, Trash2 } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import { memoryTypeLabels } from '../../types/memory';
import { formatVitalityScore } from '../../utils/memoryScores';
import { formatFullDateTime, formatListTime } from '../../utils/dateTime';
import MemoryTypeTile from './MemoryTypeTile';
import { activateOnKey } from '@/utils/activateOnKey';

interface MemoryAtomListItemProps {
  atom: MemoryAtom;
  onClick?: () => void;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
}

/** 列表视图的一行：类型图标块、标题与摘要、标签、带标签的指标与时间。 */
export default function MemoryAtomListItem({ atom, onClick, onView, onEdit, onDelete }: MemoryAtomListItemProps) {
  const typeLabel = memoryTypeLabels[atom.memory_type] || atom.memory_type;
  const updatedMs = Date.parse(atom.updated_at || atom.created_at);
  const open = () => {
    if (onClick) onClick();
    if (onView) onView(atom.id);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={atom.title}
      onClick={open}
      onKeyDown={activateOnKey(open)}
      className="group relative flex cursor-pointer items-center gap-4 border-b border-line px-4 py-3 transition-colors hover:bg-surface-container-high"
    >
      <MemoryTypeTile type={atom.memory_type} />

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <h3 className="truncate text-sm font-semibold text-fg">{atom.title}</h3>
          <span className="shrink-0 text-xs text-fg-3">{typeLabel}</span>
        </div>
        <p className="truncate text-[13px] text-fg-2">{atom.summary}</p>
        <span className="font-mono text-xs text-fg-3">{atom.alias || `mem_${atom.id.replace(/-/g, '').slice(0, 8)}`}</span>
      </div>

      <div className="hidden w-52 shrink-0 items-center gap-1.5 overflow-hidden md:flex">
        {atom.tags.slice(0, 2).map((tag) => (
          <span key={tag} className="max-w-[90px] shrink-0 truncate rounded-full bg-surface-container-high px-2 py-0.5 text-xs text-fg-2">#{tag}</span>
        ))}
        {atom.tags.length > 2 && (
          <span className="shrink-0 rounded-full bg-surface-container-high px-2 py-0.5 text-xs text-fg-3">+{atom.tags.length - 2}</span>
        )}
      </div>

      <div className="hidden shrink-0 items-center gap-4 text-xs text-fg-3 lg:flex">
        <span className="flex items-center gap-1" title="生命力：被使用越多越高，长期不用会衰减">
          <Flame className="h-3.5 w-3.5" aria-hidden="true" />
          生命力 <b className="font-semibold text-fg-2">{formatVitalityScore(atom.vitality_score)}</b>
        </span>
        <span className="flex items-center gap-1" title="置信度：生成时对内容可靠程度的判断">
          <Target className="h-3.5 w-3.5" aria-hidden="true" />
          置信度 <b className="font-semibold text-fg-2">{Math.round(atom.confidence_score * 100)}%</b>
        </span>
        {Number.isFinite(updatedMs) && (
          <span className="w-24 text-right" title={formatFullDateTime(updatedMs)}>{formatListTime(updatedMs)}</span>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        <button
          disabled
          onClick={(e) => e.stopPropagation()}
          className="cursor-not-allowed rounded-md p-1.5 text-fg-3 opacity-50"
          title="Pin 尚未提供"
          aria-label="Pin 尚未提供"
        >
          <Pin className="h-4 w-4" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); if (onEdit) onEdit(atom.id); }}
          className="rounded-md p-1.5 text-fg-3 transition-colors hover:bg-ink/10 hover:text-fg"
          title="编辑"
          aria-label="编辑"
        >
          <Edit2 className="h-4 w-4" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); if (onDelete) onDelete(atom.id); }}
          className="rounded-md p-1.5 text-fg-3 transition-colors hover:bg-magic-fire/15 hover:text-magic-fire"
          title="删除"
          aria-label="删除"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
