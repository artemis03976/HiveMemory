import { Flame } from 'lucide-react';
import type { MemoryResponse } from '@/types/memory';
import { TypeBadge } from './TypeBadge';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface MemoryListItemProps {
  memory: MemoryResponse;
  onClick: () => void;
}

export function MemoryListItem({ memory, onClick }: MemoryListItemProps) {
  return (
    <div
      className="flex items-center gap-4 px-6 py-4 border-b border-white/5 hover:bg-muted/10 cursor-pointer transition-colors"
      onClick={onClick}
    >
      {/* Type Badge */}
      <div className="shrink-0">
        <TypeBadge type={memory.memory_type} size="sm" />
      </div>

      {/* Title and Summary */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm text-foreground truncate">{memory.title}</span>
          {memory.alias && (
            <code className="text-xs text-primary font-mono shrink-0">@{memory.alias}</code>
          )}
        </div>
        <p className="text-xs text-muted-foreground truncate">{memory.summary}</p>
      </div>

      {/* Metadata */}
      <div className="flex items-center gap-6 text-xs text-muted-foreground shrink-0">
        {/* Tags count */}
        <span>{memory.tags.length} 标签</span>

        {/* Confidence */}
        <div className="w-24">
          <ConfidenceIndicator score={memory.confidence_score} />
        </div>

        {/* Access count */}
        <div className="flex items-center gap-1">
          <Flame className="w-3 h-3" />
          <span>{memory.access_count}</span>
        </div>
      </div>
    </div>
  );
}

interface MemoryListProps {
  memories: MemoryResponse[];
  onItemClick: (memory: MemoryResponse) => void;
}

export function MemoryList({ memories, onItemClick }: MemoryListProps) {
  return (
    <div className="divide-y divide-white/5">
      {memories.map((memory) => (
        <MemoryListItem key={memory.id} memory={memory} onClick={() => onItemClick(memory)} />
      ))}
    </div>
  );
}
