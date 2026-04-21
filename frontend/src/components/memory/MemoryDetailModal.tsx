import { useState } from 'react';
import { X, Copy, Pin, Edit2, Trash2, Clock, Hash, Database, FileText, AtSign, Code2, Wrench, Link as LinkIcon, User, Check, RotateCcw } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import MarkdownRenderer from '../common/MarkdownRenderer';
import { motion, AnimatePresence } from 'motion/react';
import { memoryTypeColors } from '../../types/memory';
import { useDraft } from '../../hooks/useDraft';

interface MemoryDetailModalProps {
  atom: MemoryAtom | null;
  onClose: () => void;
  onEdit: (id: string, patch: Partial<Pick<MemoryAtom, 'title' | 'summary' | 'content' | 'alias' | 'tags'>>) => Promise<void>;
  onPin: (id: string) => void;
  onDelete: (id: string) => void;
}

const typeIcons: Record<string, LucideIcon> = {
  CODE_SNIPPET: Code2,
  FACT: FileText,
  TOOL: Wrench,
  URL_RESOURCE: LinkIcon,
  REFLECTION: FileText,
  USER_PROFILE: User,
  WORK_IN_PROGRESS: Code2,
};

export default function MemoryDetailModal({ atom, onClose, onEdit, onPin, onDelete }: MemoryDetailModalProps) {
  const [editing, setEditing] = useState(false);

  const { draft, isDirty, isSaving, updateDraft, save, reset } = useDraft({
    initialData: {
      title: atom?.title || '',
      summary: atom?.summary || '',
      content: atom?.content || '',
      alias: atom?.alias || '',
      tags: atom?.tags.join(', ') || '',
    },
    onSave: async (draftData) => {
      if (!atom) return;
      await onEdit(atom.id, {
        title: draftData.title,
        summary: draftData.summary,
        content: draftData.content,
        alias: draftData.alias || null,
        tags: draftData.tags.split(',').map(t => t.trim()).filter(Boolean),
      });
    },
    onSuccess: () => setEditing(false),
  });

  if (!atom) return null;

  const typeConfig = memoryTypeColors[atom.memory_type] || { color: 'hsl(220, 10%, 50%)', name: 'unknown' };
  const TypeIcon = typeIcons[atom.memory_type] || FileText;

  const startEdit = () => {
    reset(); // 确保每次进入编辑模式时，草稿数据是最新的
    setEditing(true);
  };

  const cancelEdit = () => {
    setEditing(false);
    reset();
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-40 flex items-center justify-center p-4 sm:p-6 md:p-12 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className={`relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-surface-container-lowest border border-white/10 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden ${editing ? 'h-[85vh]' : ''}`}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between p-6 border-b border-white/5 bg-surface-dim shrink-0">
            <div className="flex-1 basis-0 min-w-0 pr-6">
              <div className={`flex ${editing ? 'items-start' : 'items-center'} gap-3 mb-1`}>
                <div
                  className={`flex items-center justify-center p-2 rounded-lg border ${editing ? 'mt-6' : ''}`}
                  style={{
                    color: typeConfig.color,
                    borderColor: typeConfig.color.replace('hsl', 'hsla').replace(')', ', 0.4)'),
                    backgroundColor: typeConfig.color.replace('hsl', 'hsla').replace(')', ', 0.2)')
                  }}
                >
                  <TypeIcon className="w-5 h-5" />
                </div>
                {editing ? (
                  <div className="flex-1 min-w-0 flex flex-col gap-1.5 w-full">
                    <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">标题</label>
                    <input
                      className="w-full text-lg font-bold bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                      placeholder="记忆标题..."
                      value={draft.title}
                      onChange={e => updateDraft({ title: e.target.value })}
                    />
                  </div>
                ) : (
                  <h2 className="flex-1 min-w-0 text-xl font-bold text-slate-100 leading-tight truncate">
                    {atom.title}
                  </h2>
                )}
                {atom.isPinned && !editing && (
                  <Pin className="w-4 h-4 text-primary fill-primary/20 shrink-0" />
                )}
              </div>

              <div className={`flex flex-col gap-2 mt-2 w-full min-w-0 ${editing ? 'pl-[52px]' : ''}`}>
                {/* Alias */}
                <div className="flex items-center gap-3 w-full min-w-0 min-h-[28px]">
                  {editing ? (
                    <div className="flex flex-col gap-1.5 w-full max-w-sm mt-2">
                      <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">别名 (可选)</label>
                      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-black/20 border border-white/10 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50 transition-all">
                        <AtSign className="w-4 h-4 text-slate-500 shrink-0" />
                        <input
                          className="flex-1 text-sm font-mono bg-transparent text-primary/90 placeholder:text-slate-600 focus:outline-none"
                          placeholder="e.g. user-preferences"
                          value={draft.alias}
                          onChange={e => updateDraft({ alias: e.target.value })}
                        />
                      </div>
                    </div>
                  ) : atom.alias ? (
                    <button
                      onClick={() => navigator.clipboard.writeText(atom.alias || '')}
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

                {/* Summary */}
                {editing ? (
                  <div className="flex flex-col gap-1.5 w-full mt-2">
                    <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">摘要</label>
                    <textarea
                      className="w-full text-sm bg-black/20 border border-white/10 rounded-lg px-3 py-2.5 text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none"
                      rows={2}
                      placeholder="简短描述这段记忆的核心内容..."
                      value={draft.summary}
                      onChange={e => updateDraft({ summary: e.target.value })}
                    />
                  </div>
                ) : (
                  <p className="w-full max-w-none text-slate-400 text-sm leading-relaxed">{atom.summary}</p>
                )}
              </div>
            </div>

            <div className="flex items-start gap-2 shrink-0">
              {editing ? (
                <div className="flex items-center gap-2 mr-2 mt-6">
                  {isDirty && (
                    <button
                      onClick={reset}
                      className="px-3 py-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 text-sm font-medium transition-colors"
                      title="重置修改"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  )}
                  <button onClick={cancelEdit} className="px-3 py-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 text-sm font-medium transition-colors">
                    取消
                  </button>
                  <button
                    onClick={save}
                    disabled={!isDirty || isSaving}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors ${
                      isDirty
                        ? 'bg-primary/10 hover:bg-primary/20 text-primary border-primary/20'
                        : 'bg-white/5 text-slate-500 border-white/5 cursor-not-allowed'
                    }`}
                  >
                    {isSaving ? (
                      <span className="w-3.5 h-3.5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                    ) : (
                      <Check className="w-3.5 h-3.5" />
                    )}
                    {isSaving ? '保存中...' : '保存'}
                  </button>
                </div>
              ) : (
                <>
                  <button onClick={() => onPin(atom.id)} className={`p-2 rounded-lg hover:bg-white/10 transition-colors ${atom.isPinned ? 'text-primary' : 'text-slate-400'}`} title="固定/取消固定">
                    <Pin className="w-4 h-4" />
                  </button>
                  <button onClick={startEdit} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="编辑">
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button onClick={() => onDelete(atom.id)} className="p-2 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors" title="删除">
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <div className="w-px h-6 bg-white/10 mx-1 mt-2" />
                  <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
                    <X className="w-5 h-5" />
                  </button>
                </>
              )}
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
                {!editing && (
                  <button
                    onClick={() => navigator.clipboard.writeText(atom.content)}
                    className="p-1.5 text-slate-500 hover:text-slate-300 hover:bg-white/5 rounded-lg transition-colors"
                    title="复制内容"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className={`flex-1 rounded-xl border ${editing ? 'bg-black/20 border-white/10 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50' : 'bg-black/40 border-white/5'} p-4 text-slate-300 text-sm leading-relaxed overflow-hidden flex flex-col transition-all`}>
                {editing ? (
                  <textarea
                    className="w-full h-full bg-transparent text-slate-300 text-sm leading-relaxed font-mono focus:outline-none resize-none placeholder:text-slate-600 scrollbar-hide"
                    value={draft.content}
                    placeholder="支持 Markdown 格式..."
                    onChange={e => updateDraft({ content: e.target.value })}
                  />
                ) : (
                  <div className="overflow-y-auto scrollbar-hide h-full">
                    <MarkdownRenderer content={atom.content} />
                  </div>
                )}
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

                <div className="p-3 rounded-xl bg-surface-container border border-white/5">
                  <div className="text-[13px] text-slate-500 mb-2 flex items-center gap-1.5">
                    <Hash className="w-3.5 h-3.5" />
                    标签
                  </div>
                  {editing ? (
                    <div className="mt-1 flex flex-col gap-1.5">
                      <input
                        className="w-full text-xs font-mono bg-black/20 border border-white/10 rounded-lg px-2.5 py-2 text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                        placeholder="e.g. react, ui, bugs"
                        value={draft.tags}
                        onChange={e => updateDraft({ tags: e.target.value })}
                      />
                      <span className="text-[10px] text-slate-500 ml-1">以逗号分隔多个标签</span>
                    </div>
                  ) : atom.tags && atom.tags.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {atom.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 rounded bg-black/20 text-xs text-slate-300 font-mono border border-white/5">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-xs text-slate-500">暂无标签</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
