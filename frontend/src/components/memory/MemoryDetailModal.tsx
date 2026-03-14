import { motion, AnimatePresence } from 'framer-motion';
import { X, Trash2, Copy, Calendar, TrendingUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import type { MemoryResponse } from '@/types/memory';
import { TypeBadge } from './TypeBadge';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { useEffect } from 'react';

interface MemoryDetailModalProps {
  memory: MemoryResponse | null;
  onClose: () => void;
  onDelete: (id: string) => Promise<void>;
}

export function MemoryDetailModal({ memory, onClose, onDelete }: MemoryDetailModalProps) {
  // Handle Esc key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  if (!memory) return null;

  const handleDelete = async () => {
    if (confirm('确定要删除这条记忆吗？此操作无法撤销。')) {
      try {
        await onDelete(memory.id);
        onClose();
      } catch (err) {
        alert('删除失败: ' + (err instanceof Error ? err.message : '未知错误'));
      }
    }
  };

  const handleCopyAlias = () => {
    if (memory.alias) {
      navigator.clipboard.writeText(`@${memory.alias}`);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          className="glass-card w-full max-w-4xl max-h-[90vh] m-4 rounded-lg overflow-hidden flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
            <div className="flex-1 min-w-0 mr-4">
              <h2 className="text-xl font-bold text-foreground mb-1">{memory.title}</h2>
              {memory.alias && (
                <div className="flex items-center gap-2">
                  <code className="text-sm text-primary font-mono">@{memory.alias}</code>
                  <button
                    onClick={handleCopyAlias}
                    className="p-1 hover:bg-white/10 rounded transition-colors"
                    title="Copy alias"
                  >
                    <Copy className="w-3.5 h-3.5 text-muted-foreground" />
                  </button>
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleDelete}
                className="p-2 hover:bg-destructive/20 rounded transition-colors"
                title="Delete memory"
              >
                <Trash2 className="w-4 h-4 text-destructive" />
              </button>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded transition-colors"
                title="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="overflow-y-auto custom-scrollbar p-6 flex-1">
            {/* Metadata Grid */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-primary/10 rounded">
                  <TypeBadge type={memory.memory_type} size="sm" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">类型</p>
                  <p className="text-sm font-medium">{memory.memory_type}</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 bg-primary/10 rounded">
                  <Calendar className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">创建时间</p>
                  <p className="text-sm font-medium">{formatDate(memory.created_at)}</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 bg-primary/10 rounded">
                  <TrendingUp className="w-4 h-4 text-primary" />
                </div>
                <div className="flex-1">
                  <p className="text-xs text-muted-foreground mb-1">置信度</p>
                  <ConfidenceIndicator score={memory.confidence_score} showLabel />
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 bg-primary/10 rounded">
                  <span className="text-sm font-bold text-primary">{memory.access_count}</span>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">访问次数</p>
                  <p className="text-sm font-medium">被引用 {memory.access_count} 次</p>
                </div>
              </div>
            </div>

            {/* Tags */}
            {memory.tags.length > 0 && (
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-foreground mb-2">标签</h4>
                <div className="flex flex-wrap gap-2">
                  {memory.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 rounded-full bg-primary/20 text-primary text-sm"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Summary */}
            <div className="mb-6">
              <h4 className="text-sm font-semibold text-foreground mb-2">摘要</h4>
              <p className="text-sm text-muted-foreground">{memory.summary}</p>
            </div>

            {/* Markdown Content */}
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-foreground mb-2">完整内容</h4>
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                  {memory.content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
