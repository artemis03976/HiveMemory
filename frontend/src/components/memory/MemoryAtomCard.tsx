import { Clock, BrainCircuit, Pin, Flame, Edit2, Trash2, AtSign } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import { memoryTypeColors, memoryTypeLabels } from '../../types/memory';
import { formatVitalityScore, isHighVitality } from '../../utils/memoryScores';

interface MemoryAtomCardProps {
  atom: MemoryAtom;
  onClick?: () => void;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onPin?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export default function MemoryAtomCard({ atom, onClick, onView, onEdit, onPin, onDelete }: MemoryAtomCardProps) {
  const typeConfig = memoryTypeColors[atom.memory_type] || { color: 'hsl(220, 10%, 50%)', name: 'unknown' };
  const typeLabel = memoryTypeLabels[atom.memory_type] || atom.memory_type;
  const highVitality = isHighVitality(atom.vitality_score);
  
  return (
    <div 
      onClick={() => {
        if (onClick) onClick();
        if (onView) onView(atom.id);
      }}
      className="group relative flex flex-col p-5 pb-4 rounded-2xl bg-surface-container border border-white/5 hover:bg-surface-container-high transition-all duration-300 cursor-pointer overflow-hidden h-64"
    >
      
      <div className="relative flex-1 flex flex-col">
        {/* 头部信息 */}
        <div className="flex items-start justify-between gap-4 mb-3 shrink-0">
          {/* 类型图标/标识 */}
          <div className="flex items-center gap-2">
            <div 
              className="w-2 h-2 rounded-full shadow-[0_0_8px_currentColor]"
              style={{ backgroundColor: typeConfig.color, color: typeConfig.color }}
            />
            <span className="text-xs font-bold tracking-wider uppercase text-slate-300">
              {typeLabel}
            </span>
          </div>

          {/* 操作按钮 */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity absolute right-0 top-0">
            <button 
              onClick={(e) => { e.stopPropagation(); if(onPin) onPin(atom.id); }} 
              className={`p-1.5 rounded hover:bg-white/10 transition-colors ${atom.isPinned ? 'text-primary' : 'text-slate-400'}`} 
              title="固定/取消固定"
            >
              <Pin className="w-3.5 h-3.5" />
            </button>
            <button 
              onClick={(e) => { e.stopPropagation(); if(onEdit) onEdit(atom.id); }} 
              className="p-1.5 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors" 
              title="编辑"
            >
              <Edit2 className="w-3.5 h-3.5" />
            </button>
            <button 
              onClick={(e) => { e.stopPropagation(); if(onDelete) onDelete(atom.id); }} 
              className="p-1.5 rounded hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors" 
              title="删除"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
          {atom.isPinned && <Pin className="w-3.5 h-3.5 text-primary absolute top-1 right-1 group-hover:hidden" />}
        </div>

        {/* 标题与摘要 */}
        <h3 className="text-base font-bold text-slate-100 mb-1.5 line-clamp-1 group-hover:text-white transition-colors shrink-0">
          {atom.title}
        </h3>
        
        {/* 别名 */}
        <div className="flex items-center gap-1.5 mb-2 shrink-0 h-5 ">
          {atom.alias ? (
            <>
              <AtSign className="w-3.5 h-3.5 text-primary/70" />
              <span className="text-xs font-mono text-primary/80 truncate">{atom.alias}</span>
            </>
          ) : (
            <>
              <AtSign className="w-3.5 h-3.5 text-slate-600" />
              <span className="text-xs font-mono text-slate-500">暂无别名</span>
            </>
          )}
        </div>

        <p className="text-sm text-slate-400 line-clamp-2 mb-4">
          {atom.summary}
        </p>

        {/* 标签 */}
        <div className="flex flex-wrap gap-1.5 mb-4 shrink-0 mt-auto">
          {atom.tags.slice(0, 3).map(tag => (
            <span 
              key={tag} 
              className="px-2 py-0.5 rounded-md bg-black/20 border border-white/5 text-xs text-slate-400 font-mono"
            >
              #{tag}
            </span>
          ))}
          {atom.tags.length > 3 && (
            <span className="px-2 py-0.5 rounded-md bg-black/20 border border-white/5 text-xs text-slate-500 font-mono">
              +{atom.tags.length - 3}
            </span>
          )}
        </div>

        {/* 底部统计信息 */}
        <div className="flex items-center justify-between pt-3 border-t border-white/5 text-xs font-mono text-slate-500 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1" title="活力值 (Vitality Score)">
              <Flame className={`w-3.5 h-3.5 ${highVitality ? 'text-amber-400' : ''}`} />
              {formatVitalityScore(atom.vitality_score)}
            </div>
            <div className="flex items-center gap-1" title="置信度 (Confidence)">
              <BrainCircuit className="w-3.5 h-3.5" />
              {atom.confidence_score.toFixed(2)}
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            {new Date(atom.created_at).toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
}
