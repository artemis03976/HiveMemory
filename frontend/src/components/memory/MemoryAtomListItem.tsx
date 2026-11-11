import { Clock, BrainCircuit, Pin, Flame, Edit2, Trash2, AtSign } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import { memoryTypeColors, memoryTypeLabels } from '../../types/memory';
import { formatVitalityScore, isHighVitality } from '../../utils/memoryScores';

interface MemoryAtomListItemProps {
  atom: MemoryAtom;
  onClick?: () => void;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onPin?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export default function MemoryAtomListItem({ atom, onClick, onView, onEdit, onPin, onDelete }: MemoryAtomListItemProps) {
  const typeConfig = memoryTypeColors[atom.memory_type] || { color: 'hsl(220, 10%, 50%)', name: 'unknown' };
  const typeLabel = memoryTypeLabels[atom.memory_type] || atom.memory_type;
  const highVitality = isHighVitality(atom.vitality_score);
  
  return (
    <div 
      onClick={() => {
        if (onClick) onClick();
        if (onView) onView(atom.id);
      }}
      className="group relative flex items-center gap-6 p-4 border-b border-line bg-surface/30 hover:bg-surface-container-high transition-all duration-300 cursor-pointer overflow-hidden"
    >
      
      {/* 类型图标/标识 */}
      <div className="flex flex-col items-center justify-center shrink-0 w-16 gap-1.5">
        <div 
          className="w-8 h-8 rounded-xl flex items-center justify-center bg-surface-container shadow-inner border border-line"
        >
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: typeConfig.color, color: typeConfig.color }}
          />
        </div>
        <span className="text-xs font-bold text-fg-3 group-hover:text-fg-2 transition-colors">
          {typeLabel}
        </span>
      </div>

      {/* 主要内容 */}
      <div className="flex-1 min-w-0 py-1">
        <div className="flex items-center gap-3 mb-1">
          <h3 className="text-base font-bold text-fg truncate group-hover:text-fg transition-colors">
            {atom.title}
          </h3>
          {atom.isPinned && <Pin className="w-3.5 h-3.5 text-primary fill-primary/20 shrink-0" />}
        </div>
        
        {/* 别名 */}
        <div className="flex items-center gap-1.5 mb-1.5">
          {atom.alias ? (
            <>
              <AtSign className="w-3.5 h-3.5 text-primary/70" />
              <span className="text-xs font-mono text-primary/80 truncate">{atom.alias}</span>
            </>
          ) : (
            <>
              <AtSign className="w-3.5 h-3.5 text-fg-3" />
              <span className="text-xs font-mono text-fg-3">暂无别名</span>
            </>
          )}
        </div>

        <p className="text-sm text-fg-3 truncate pr-8">
          {atom.summary}
        </p>
      </div>

      {/* 标签 */}
      <div className="hidden md:flex items-center gap-1.5 w-56 shrink-0 overflow-hidden">
        {atom.tags.slice(0, 2).map(tag => (
          <span 
            key={tag} 
            className="px-2 py-0.5 rounded-md bg-field border border-line text-xs text-fg-3 font-mono truncate max-w-[80px] shrink-0"
          >
            #{tag}
          </span>
        ))}
        {atom.tags.length > 2 && (
          <span className="px-2 py-0.5 rounded-md bg-field border border-line text-xs text-fg-3 font-mono shrink-0">
            +{atom.tags.length - 2}
          </span>
        )}
      </div>

      {/* 统计信息 */}
      <div className="hidden lg:flex items-center gap-6 shrink-0 text-xs font-mono text-fg-3 w-64 justify-end pr-4">
        <div className="flex items-center gap-1.5" title="活力值 (Vitality Score)">
          <Flame className={`w-4 h-4 ${highVitality ? 'text-magic-metal' : ''}`} />
          <span className={highVitality ? 'text-magic-metal/80 font-bold' : ''}>
            {formatVitalityScore(atom.vitality_score)}
          </span>
        </div>
        <div className="flex items-center gap-1.5" title="置信度 (Confidence)">
          <BrainCircuit className="w-4 h-4" />
          {atom.confidence_score.toFixed(2)}
        </div>
        <div className="flex items-center gap-1.5 w-16 justify-end">
          <Clock className="w-3.5 h-3.5" />
          {new Date(atom.created_at).toLocaleDateString()}
        </div>
      </div>

      {/* 操作 */}
      <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity pr-2">
        <button 
          onClick={(e) => { e.stopPropagation(); if(onPin) onPin(atom.id); }} 
          className={`p-1.5 rounded-lg hover:bg-ink/10 transition-colors ${atom.isPinned ? 'text-primary' : 'text-fg-3'}`}
          title="固定/取消固定"
        >
          <Pin className="w-4 h-4" />
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); if(onEdit) onEdit(atom.id); }} 
          className="p-1.5 rounded-lg hover:bg-ink/10 text-fg-3 hover:text-fg transition-colors"
          title="编辑"
        >
          <Edit2 className="w-4 h-4" />
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); if(onDelete) onDelete(atom.id); }} 
          className="p-1.5 rounded-lg hover:bg-magic-fire/20 text-fg-3 hover:text-magic-fire transition-colors"
          title="删除"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
