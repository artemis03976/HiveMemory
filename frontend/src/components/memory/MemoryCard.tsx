import { motion } from 'framer-motion';
import { Flame, Copy } from 'lucide-react';
import type { MemoryResponse } from '@/types/memory';
import { TypeBadge } from './TypeBadge';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface MemoryCardProps {
  memory: MemoryResponse;
  onClick: () => void;
}

export function MemoryCard({ memory, onClick }: MemoryCardProps) {
  const handleCopyAlias = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (memory.alias) {
      navigator.clipboard.writeText(`@${memory.alias}`);
    }
  };

  return (
    <motion.div
      className="glass-card p-4 rounded-lg cursor-pointer flex flex-col h-full min-h-[240px]"
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3 gap-2">
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-foreground mb-1 line-clamp-2">
            {memory.title}
          </h3>
          <div className="h-5 flex items-center">
            {memory.alias ? (
              <div className="flex items-center gap-1">
                <code className="text-xs font-mono text-primary">@{memory.alias}</code>
                <button
                  onClick={handleCopyAlias}
                  className="p-0.5 hover:bg-white/10 rounded transition-colors"
                  title="Copy alias"
                >
                  <Copy className="w-3 h-3 text-muted-foreground" />
                </button>
              </div>
            ) : (
              <span className="text-xs text-muted-foreground/50">无别名</span>
            )}
          </div>
        </div>
        <TypeBadge type={memory.memory_type} size="sm" />
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-3 min-h-[24px]">
        {memory.tags.length > 0 ? (
          <>
            {memory.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-xs"
              >
                #{tag}
              </span>
            ))}
            {memory.tags.length > 3 && (
              <span className="text-xs text-muted-foreground px-2 py-0.5">
                +{memory.tags.length - 3}
              </span>
            )}
          </>
        ) : null}
      </div>

      {/* Summary */}
      <p className="text-sm text-muted-foreground line-clamp-2 mb-3 flex-1">
        {memory.summary}
      </p>

      {/* Metrics */}
      <div className="flex items-center justify-between gap-4 mt-auto">
        <div className="flex-1">
          <ConfidenceIndicator score={memory.confidence_score} />
        </div>
        <div className="flex items-center gap-1 text-muted-foreground">
          <Flame className="w-3 h-3" />
          <span className="text-xs font-medium">{memory.access_count}</span>
        </div>
      </div>
    </motion.div>
  );
}
