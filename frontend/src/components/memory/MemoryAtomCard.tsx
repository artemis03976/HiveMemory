import { Flame, Target, Pin, Edit2, Trash2 } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import { memoryTypeLabels } from '../../types/memory';
import { formatVitalityScore } from '../../utils/memoryScores';
import { formatFullDateTime, formatListTime } from '../../utils/dateTime';
import MemoryTypeTile from './MemoryTypeTile';

interface MemoryAtomCardProps {
  atom: MemoryAtom;
  onClick?: () => void;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
}

/** 记忆卡片：高度随内容，正文限制行数；底部的指标与时间始终可见。 */
export default function MemoryAtomCard({ atom, onClick, onView, onEdit, onDelete }: MemoryAtomCardProps) {
  const typeLabel = memoryTypeLabels[atom.memory_type] || atom.memory_type;
  const updatedMs = Date.parse(atom.updated_at || atom.created_at);

  return (
    <div
      onClick={() => {
        if (onClick) onClick();
        if (onView) onView(atom.id);
      }}
      className="group relative flex min-w-0 cursor-pointer flex-col gap-2.5 rounded-2xl border border-line bg-surface-container p-4 transition-all duration-200 hover:-translate-y-px hover:border-primary/40"
    >
      <div className="flex min-w-0 items-center gap-2.5 pr-20">
        <MemoryTypeTile type={atom.memory_type} />
        <span className="flex min-w-0 flex-col">
          <span className="text-xs font-semibold text-fg-2">{typeLabel}</span>
          <span className="truncate font-mono text-xs text-fg-3" title={atom.alias ? undefined : '暂无别名，显示 ID 前缀'}>
            {atom.alias || `mem_${atom.id.replace(/-/g, '').slice(0, 8)}`}
          </span>
        </span>
      </div>

      <div className="absolute right-3 top-3 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        <button
          disabled
          onClick={(e) => e.stopPropagation()}
          className="cursor-not-allowed rounded-md p-1.5 text-fg-3 opacity-50"
          title="Pin 尚未提供"
          aria-label="Pin 尚未提供"
        >
          <Pin className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); if (onEdit) onEdit(atom.id); }}
          className="rounded-md p-1.5 text-fg-3 transition-colors hover:bg-ink/10 hover:text-fg"
          title="编辑"
          aria-label="编辑"
        >
          <Edit2 className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); if (onDelete) onDelete(atom.id); }}
          className="rounded-md p-1.5 text-fg-3 transition-colors hover:bg-magic-fire/15 hover:text-magic-fire"
          title="删除"
          aria-label="删除"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      <h3 className="line-clamp-2 text-base font-semibold leading-snug text-fg">{atom.title}</h3>
      <p className="line-clamp-3 text-[13px] leading-relaxed text-fg-2">{atom.summary}</p>

      {atom.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {atom.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="rounded-full bg-surface-container-high px-2 py-0.5 text-xs text-fg-2">#{tag}</span>
          ))}
          {atom.tags.length > 3 && (
            <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-xs text-fg-3">+{atom.tags.length - 3}</span>
          )}
        </div>
      )}

      <div className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-2.5 text-xs text-fg-3">
        <span className="flex items-center gap-1 whitespace-nowrap" title="生命力：被使用越多越高，长期不用会衰减">
          <Flame className="h-3.5 w-3.5" aria-hidden="true" />
          生命力 <b className="font-semibold text-fg-2">{formatVitalityScore(atom.vitality_score)}</b>
        </span>
        <span className="flex items-center gap-1 whitespace-nowrap" title="置信度：生成时对内容可靠程度的判断">
          <Target className="h-3.5 w-3.5" aria-hidden="true" />
          置信度 <b className="font-semibold text-fg-2">{Math.round(atom.confidence_score * 100)}%</b>
        </span>
        {Number.isFinite(updatedMs) && (
          <span className="ml-auto whitespace-nowrap" title={formatFullDateTime(updatedMs)}>{formatListTime(updatedMs)}</span>
        )}
      </div>
    </div>
  );
}
