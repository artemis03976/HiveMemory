import { useMemo, useState } from 'react';
import { X, Copy, Pin, Edit2, Trash2, Clock, Hash, Database, FileText, AtSign, Check, RotateCcw, Globe, Lock } from 'lucide-react';
import type { MemoryAtom } from '../../types/memory';
import MarkdownRenderer from '../common/MarkdownRenderer';
import { motion, AnimatePresence } from 'motion/react';
import { memoryTypeLabels } from '../../types/memory';
import MemoryTypeTile from './MemoryTypeTile';
import { formatDateTime, formatFullDateTime } from '../../utils/dateTime';
import { useDraft } from '../../hooks/useDraft';
import { useDialog } from '../../hooks/useDialog';
import { formatVitalityScore } from '../../utils/memoryScores';
import type { VisibilityAgents } from '@/hooks/useVisibilityAgents';
import { isListedAgent, policyTarget, targetPolicy, visibilityAgentLabel, VISIBILITY_NOTE } from '@/utils/memoryVisibility';
import MemoryVisibilitySelect from './MemoryVisibilitySelect';

interface MemoryDetailModalProps {
  atom: MemoryAtom | null;
  onClose: () => void;
  onEdit: (id: string, patch: Partial<Pick<MemoryAtom, 'title' | 'summary' | 'content' | 'alias' | 'tags' | 'access_policy'>>) => Promise<void>;
  onDelete: (id: string) => void;
  visibilityAgents: VisibilityAgents;
}

/** 元数据栏里的一行：名称、可选的短条与数值，悬停说明含义。 */
function MetaRow({ label, value, ratio, hint }: { label: string; value: string; ratio?: number; hint?: string }) {
  return (
    <div className="flex items-center gap-2 text-[13px]" title={hint}>
      <span className="flex-1 text-fg-2">{label}</span>
      {ratio !== undefined && (
        <span aria-hidden="true" className="h-[5px] w-[70px] overflow-hidden rounded-full bg-surface-container-high">
          <span className="block h-full rounded-full bg-primary" style={{ width: `${Math.max(0, Math.min(1, ratio)) * 100}%` }} />
        </span>
      )}
      <span className="min-w-9 text-right font-mono text-xs text-fg">{value}</span>
    </div>
  );
}

export default function MemoryDetailModal({ atom, onClose, onEdit, onDelete, visibilityAgents }: MemoryDetailModalProps) {
  const [editing, setEditing] = useState(false);
  const savedTarget = atom ? policyTarget(atom.access_policy) : '';
  const initialDraftData = useMemo(() => ({
    title: atom?.title || '',
    summary: atom?.summary || '',
    content: atom?.content || '',
    alias: atom?.alias || '',
    tags: atom?.tags.join(', ') || '',
    visibilityTarget: atom ? policyTarget(atom.access_policy) : '',
  }), [atom]);

  const { draft, isDirty, isSaving, updateDraft, save, reset } = useDraft({
    initialData: initialDraftData,
    onSave: async (draftData) => {
      if (!atom) return;
      await onEdit(atom.id, {
        title: draftData.title,
        summary: draftData.summary,
        content: draftData.content,
        alias: draftData.alias || null,
        tags: draftData.tags.split(',').map(t => t.trim()).filter(Boolean),
        // 策略单独写入、不产生新版本；没有改动时不提交，避免多一次写入。
        ...(draftData.visibilityTarget !== savedTarget
          ? { access_policy: targetPolicy(draftData.visibilityTarget) }
          : {}),
      });
    },
    onSuccess: () => setEditing(false),
  });
  // Esc 等同于当前可见的关闭操作：编辑时是“取消”，否则关闭对话框
  const { panelRef, onKeyDown } = useDialog<HTMLDivElement>(atom !== null, () => {
    if (!editing) return onClose();
    setEditing(false);
    reset();
  });

  if (!atom) return null;
  // 只有候选列表读全时才能断定目标已不在列表中。
  const targetUnlisted = savedTarget !== '' && visibilityAgents.complete && !isListedAgent(visibilityAgents.groups, savedTarget);


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
        className="fixed inset-0 z-40 flex items-center justify-center p-4 sm:p-6 md:p-12 bg-scrim backdrop-blur-sm"
        onMouseDown={(e) => {
          if (e.target === e.currentTarget) {
            onClose();
          }
        }}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-label={atom.title}
          tabIndex={-1}
          onKeyDown={onKeyDown}
          className={`relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-surface-container-lowest border border-line rounded-2xl shadow-pop overflow-hidden outline-none ${editing ? 'h-[85vh]' : ''}`}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between p-6 border-b border-line bg-surface-dim shrink-0">
            <div className="flex-1 basis-0 min-w-0 pr-6">
              <div className={`flex ${editing ? 'items-start' : 'items-center'} gap-3 mb-1`}>
                <div className={editing ? 'mt-6' : ''} title={memoryTypeLabels[atom.memory_type] ?? atom.memory_type}>
                  <MemoryTypeTile type={atom.memory_type} size="lg" />
                </div>
                {editing ? (
                  <div className="flex-1 min-w-0 flex flex-col gap-1.5 w-full">
                    <label className="text-xs text-fg-3 font-medium ml-1">标题</label>
                    <input
                      className="w-full text-lg font-bold bg-field border border-line rounded-lg px-3 py-2 text-fg placeholder:text-fg-3 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                      placeholder="记忆标题..."
                      value={draft.title}
                      onChange={e => updateDraft({ title: e.target.value })}
                    />
                  </div>
                ) : (
                  <h2 className="flex-1 min-w-0 text-xl font-bold text-fg leading-tight truncate">
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
                      <label className="text-xs text-fg-3 font-medium ml-1">别名 (可选)</label>
                      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-field border border-line focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50 transition-all">
                        <AtSign className="w-4 h-4 text-fg-3 shrink-0" />
                        <input
                          className="flex-1 text-sm font-mono bg-transparent text-primary/90 placeholder:text-fg-3 focus:outline-none"
                          placeholder="e.g. user-preferences"
                          value={draft.alias}
                          onChange={e => updateDraft({ alias: e.target.value })}
                        />
                      </div>
                    </div>
                  ) : atom.alias ? (
                    <button
                      onClick={() => navigator.clipboard.writeText(atom.alias || '')}
                      className="flex items-center gap-1.5 px-2 py-1 rounded bg-field hover:bg-ink/10 border border-line transition-colors group"
                    >
                      <AtSign className="w-3.5 h-3.5 text-primary/70 group-hover:text-primary" />
                      <span className="text-xs font-mono text-primary/90 group-hover:text-primary">{atom.alias}</span>
                      <Copy className="w-3 h-3 text-fg-3 group-hover:text-fg-2 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  ) : (
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-field border border-line">
                      <AtSign className="w-3.5 h-3.5 text-fg-3" />
                      <span className="text-xs font-mono text-fg-3">暂无别名</span>
                    </div>
                  )}
                </div>

                {/* Summary */}
                {editing ? (
                  <div className="flex flex-col gap-1.5 w-full mt-2">
                    <label className="text-xs text-fg-3 font-medium ml-1">摘要</label>
                    <textarea
                      className="w-full text-sm bg-field border border-line rounded-lg px-3 py-2.5 text-fg-2 placeholder:text-fg-3 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none"
                      rows={2}
                      placeholder="简短描述这段记忆的核心内容..."
                      value={draft.summary}
                      onChange={e => updateDraft({ summary: e.target.value })}
                    />
                  </div>
                ) : (
                  <p className="w-full max-w-none text-fg-3 text-sm leading-relaxed">{atom.summary}</p>
                )}
              </div>
            </div>

            <div className="flex items-start gap-2 shrink-0">
              {editing ? (
                <div className="flex items-center gap-2 mr-2 mt-6">
                  {isDirty && (
                    <button
                      onClick={reset}
                      className="px-3 py-1.5 rounded-lg hover:bg-ink/5 text-fg-3 hover:text-fg text-sm font-medium transition-colors"
                      title="重置修改"
                      aria-label="重置修改"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  )}
                  <button onClick={cancelEdit} className="px-3 py-1.5 rounded-lg hover:bg-ink/5 text-fg-3 hover:text-fg text-sm font-medium transition-colors">
                    取消
                  </button>
                  <button
                    onClick={save}
                    disabled={!isDirty || isSaving}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors ${
                      isDirty
                        ? 'bg-primary/10 hover:bg-primary/20 text-primary border-primary/20'
                        : 'bg-ink/5 text-fg-3 border-line cursor-not-allowed'
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
                  <button onClick={startEdit} className="p-2 rounded-lg hover:bg-ink/10 text-fg-3 hover:text-fg transition-colors" title="编辑" aria-label="编辑">
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button disabled className="cursor-not-allowed rounded-lg p-2 text-fg-3 opacity-45" title="Pin 尚未提供" aria-label="Pin 尚未提供">
                    <Pin className="w-4 h-4" />
                  </button>
                  <div className="w-px h-6 bg-ink/10 mx-1 mt-2" />
                  <button onClick={() => onDelete(atom.id)} className="p-2 rounded-lg hover:bg-magic-fire/20 text-fg-3 hover:text-magic-fire transition-colors" title="删除这条记忆" aria-label="删除这条记忆">
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <div className="w-px h-6 bg-ink/10 mx-1 mt-2" />
                  <button onClick={onClose} className="p-2 rounded-lg hover:bg-ink/10 text-fg-3 hover:text-fg transition-colors" title="关闭" aria-label="关闭">
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
                <h3 className="text-sm font-bold text-fg-2 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  记忆内容
                </h3>
                {!editing && (
                  <button
                    onClick={() => navigator.clipboard.writeText(atom.content)}
                    className="p-1.5 text-fg-3 hover:text-fg-2 hover:bg-ink/5 rounded-lg transition-colors"
                    title="复制内容"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className={`flex-1 rounded-xl border ${editing ? 'bg-field border-line focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50' : 'bg-field border-line'} p-4 text-fg-2 text-sm leading-relaxed overflow-hidden flex flex-col transition-all`}>
                {editing ? (
                  <textarea
                    className="w-full h-full bg-transparent text-fg-2 text-sm leading-relaxed font-mono focus:outline-none resize-none placeholder:text-fg-3 scrollbar-hide"
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
                <div className="p-3 rounded-xl bg-surface-container border border-line">
                  <div className="text-[13px] text-fg-3 mb-2 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5" />
                    元数据
                  </div>
                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2 text-[13px]">
                      <span className="flex-1 text-fg-2">ID</span>
                      <button
                        onClick={() => navigator.clipboard.writeText(atom.id)}
                        className="flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-xs text-fg-2 hover:bg-ink/5"
                        title="复制完整 ID"
                        aria-label="复制完整 ID"
                      >
                        {atom.id.substring(0, 8)}…
                        <Copy className="h-3 w-3" />
                      </button>
                    </div>
                    <MetaRow label="置信度" value={`${Math.round(atom.confidence_score * 100)}%`} ratio={atom.confidence_score} hint="生成时对内容可靠程度的判断" />
                    <MetaRow label="生命力" value={formatVitalityScore(atom.vitality_score)} ratio={atom.vitality_score / 100} hint="被使用越多越高，长期不用会衰减；低于阈值时自动归档" />
                    <MetaRow label="被引用" value={`${atom.access_count} 次`} />
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-surface-container border border-line">
                  <div className="text-[13px] text-fg-3 mb-2 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    时间线
                  </div>
                  <div className="space-y-2.5 text-[13px]">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-fg-2">创建于</span>
                      <span className="text-fg" title={formatFullDateTime(Date.parse(atom.created_at))}>{formatDateTime(Date.parse(atom.created_at))}</span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-fg-2">最近更新</span>
                      <span className="text-fg" title={formatFullDateTime(Date.parse(atom.updated_at))}>{formatDateTime(Date.parse(atom.updated_at))}</span>
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-surface-container border border-line">
                  <div className="text-[13px] text-fg-3 mb-2 flex items-center gap-1.5">
                    <Hash className="w-3.5 h-3.5" />
                    标签
                  </div>
                  {editing ? (
                    <div className="mt-1 flex flex-col gap-1.5">
                      <input
                        className="w-full text-xs font-mono bg-field border border-line rounded-lg px-2.5 py-2 text-fg-2 placeholder:text-fg-3 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                        placeholder="e.g. react, ui, bugs"
                        value={draft.tags}
                        onChange={e => updateDraft({ tags: e.target.value })}
                      />
                      <span className="text-xs text-fg-3 ml-1">以逗号分隔多个标签</span>
                    </div>
                  ) : atom.tags && atom.tags.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {atom.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 rounded bg-field text-xs text-fg-2 font-mono border border-line">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-xs text-fg-3">暂无标签</span>
                  )}
                </div>

                <div className="p-3 rounded-xl bg-surface-container border border-line">
                  <div className="text-[13px] text-fg-3 mb-2 flex items-center gap-1.5">
                    <Lock className="w-3.5 h-3.5" />
                    可见范围
                  </div>
                  {editing ? (
                    <div className="mt-1 flex flex-col gap-1.5">
                      <MemoryVisibilitySelect
                        ariaLabel="可见范围"
                        value={draft.visibilityTarget}
                        onChange={visibilityTarget => updateDraft({ visibilityTarget })}
                        agents={visibilityAgents}
                      />
                      <span className="text-xs text-fg-3 ml-1" title={VISIBILITY_NOTE}>只限定 Agent 的检索与读取，不是保密</span>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-1">
                      <span className={`flex items-center gap-1.5 text-[13px] font-semibold ${targetUnlisted ? 'text-magic-metal' : 'text-fg'}`}>
                        {savedTarget ? <Lock className="h-3.5 w-3.5" aria-hidden="true" /> : <Globe className="h-3.5 w-3.5" aria-hidden="true" />}
                        {savedTarget ? <span className="font-mono">{visibilityAgentLabel(savedTarget)}</span> : '所有 Agent'}
                      </span>
                      {!targetUnlisted && (
                        <span className="text-xs leading-relaxed text-fg-3">
                          {savedTarget ? '只有这个 Agent 能检索与读取，管理页照常可见' : '接入的 Agent 都能检索与读取'}
                        </span>
                      )}
                      {targetUnlisted && (
                        <span role="note" className="mt-1 rounded-lg bg-magic-metal/12 px-2.5 py-2 text-xs leading-relaxed text-magic-metal">
                          这个 Agent 已不在 Agent 列表中，现在没有 Agent 能检索到这条记忆。编辑时可以改为其他 Agent 或所有 Agent。
                        </span>
                      )}
                    </div>
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
