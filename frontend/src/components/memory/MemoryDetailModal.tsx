import { X, Copy, Pin, Edit2, Trash2, Clock, Hash, Database, FileText, AtSign, Code2, Wrench, Link as LinkIcon, User } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import MarkdownRenderer from '../common/MarkdownRenderer';
import { motion, AnimatePresence } from 'motion/react';
import { memoryTypeColors } from '../../types/memory';

interface MemoryDetailModalProps {
  atom: MemoryAtom | null;
  onClose: () => void;
  onEdit: (id: string) => void;
  onPin: (id: string) => void;
  onDelete: (id: string) => void;
}

const typeIcons: Record<string, any> = {
  CODE_SNIPPET: Code2,
  FACT: FileText,
  TOOL: Wrench,
  URL_RESOURCE: LinkIcon,
  REFLECTION: FileText,
  USER_PROFILE: User,
  WORK_IN_PROGRESS: Code2,
};

export default function MemoryDetailModal({ atom, onClose, onEdit, onPin, onDelete }: MemoryDetailModalProps) {
  if (!atom) return null;
  
  const typeConfig = memoryTypeColors[atom.memory_type] || { color: 'hsl(220, 10%, 50%)', name: 'unknown' };
  const TypeIcon = typeIcons[atom.memory_type] || FileText;

  const handleCopyAlias = () => {
    navigator.clipboard.writeText(atom.alias || '');
  };

  const handleCopyPayload = () => {
    navigator.clipboard.writeText(atom.content);
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-100 flex items-center justify-center p-4 sm:p-6 md:p-12 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-surface-container-lowest border border-white/10 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between p-6 border-b border-white/5 bg-surface-dim shrink-0">
            <div className="flex-1 basis-0 min-w-0 pr-6">
              <div className="flex items-center gap-3 mb-1">
                <div 
                  className="flex items-center justify-center p-2 rounded-lg border"
                  style={{ 
                    color: typeConfig.color, 
                    borderColor: typeConfig.color.replace('hsl', 'hsla').replace(')', ', 0.4)'),
                    backgroundColor: typeConfig.color.replace('hsl', 'hsla').replace(')', ', 0.2)')
                  }}
                >
                  <TypeIcon className={`w-5 h-5`} />
                </div>
                <h2 className="flex-1 min-w-0 text-xl font-bold text-slate-100 leading-tight truncate">
                  {atom.title}
                </h2>
                {atom.isPinned && (
                  <Pin className="w-4 h-4 text-primary fill-primary/20 shrink-0" />
                )}
              </div>
              
              <div className="flex flex-col gap-2 mt-2 w-full min-w-0">
                <div className="flex items-center gap-3 w-full min-w-0 min-h-[28px]">
                  {atom.alias ? (
                    <button 
                      onClick={handleCopyAlias}
                      className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/30 hover:bg-black/50 border border-white/5 transition-colors group"
                    >
                      <AtSign className="w-3.5 h-3.5 text-primary/70 group-hover:text-primary" />
                      <span className="text-xs font-mono text-primary/90 group-hover:text-primary">{atom.alias}</span>
                      <Copy className="w-3 h-3 text-slate-500 group-hover:text-slate-300 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  ) : (
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/10 border border-white/5">
                      <AtSign className="w-3.5 h-3.5 text-slate-600" />
                      <span className="text-xs font-mono text-slate-500">暂无别名</span>
                    </div>
                  )}
                </div>

                <p className="w-full max-w-none text-slate-400 text-sm leading-relaxed">
                  {atom.summary}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button onClick={() => onPin(atom.id)} className={`p-2 rounded-lg hover:bg-white/10 transition-colors ${atom.isPinned ? 'text-primary' : 'text-slate-400'}`} title="固定/取消固定">
                <Pin className="w-4 h-4" />
              </button>
              <button onClick={() => onEdit(atom.id)} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="编辑">
                <Edit2 className="w-4 h-4" />
              </button>
              <button onClick={() => onDelete(atom.id)} className="p-2 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors" title="删除">
                <Trash2 className="w-4 h-4" />
              </button>
              <div className="w-px h-6 bg-white/10 mx-1" />
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-hidden p-6 flex gap-6">
            {/* 主要内容 */}
            <div className="flex-1 min-w-0 flex flex-col">
              <div className="flex items-center justify-between mb-4 shrink-0">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  记忆内容
                </h3>
                <button 
                  onClick={handleCopyPayload}
                  className="p-1.5 text-slate-500 hover:text-slate-300 hover:bg-white/5 rounded-lg transition-colors"
                  title="复制内容"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 bg-black/40 rounded-xl border border-white/5 p-4 text-slate-300 text-sm leading-relaxed overflow-y-auto scrollbar-hide">
                <MarkdownRenderer content={atom.content} />
              </div>
            </div>

            {/* 侧边栏元数据 */}
            <div className="w-64 shrink-0 flex flex-col gap-6 overflow-y-auto scrollbar-hide pr-1">
              {/* 元数据卡片 */} 
              <div className="flex flex-col gap-3">
                <div className="p-3 rounded-xl bg-surface-container border border-white/5">
                  <div className="text-[13px] text-slate-500 mb-2 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5" />
                    元数据
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">ID</span>
                      <span className="text-slate-200 font-mono text-xs">{atom.id.substring(0, 8)}...</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">置信度</span>
                      <span className="text-primary font-mono">{atom.confidence_score.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">生命力分数</span>
                      <span className="text-amber-400 font-mono">{(atom.vitality_score * 100).toFixed(0)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">访问次数</span>
                      <span className="text-slate-200 font-mono">{atom.access_count}</span>
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-surface-container border border-white/5">
                  <div className="text-[13px] text-slate-500 mb-2 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    时间线
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">创建于</span>
                      <span className="text-slate-200">{new Date(atom.created_at).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">最后更新</span>
                      <span className="text-slate-200">{new Date(atom.updated_at).toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                {atom.tags && atom.tags.length > 0 && (
                  <div className="p-3 rounded-xl bg-surface-container border border-white/5">
                    <div className="text-[13px] text-slate-500 mb-2 flex items-center gap-1.5">
                      <Hash className="w-3.5 h-3.5" />
                      标签
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {atom.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 rounded bg-black/20 text-xs text-slate-300 font-mono border border-white/5">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
