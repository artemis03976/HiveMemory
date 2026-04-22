import { useState } from 'react';
import { Bot, Power, Save, Hash, Trash2, RotateCcw, Check } from 'lucide-react';
import type { AgentData } from '@/types/agent';
import { ConfirmDialog } from '../common/ConfirmDialog';

interface AgentEditorHeaderProps {
  agent: AgentData;
  onUpdate: (updates: Partial<AgentData>) => void;
  onSave: () => void;
  onDelete: () => void;
  isDirty?: boolean;
  isSaving?: boolean;
  onReset?: () => void;
}

export function AgentEditorHeader({ agent, onUpdate, onSave, onDelete, isDirty, isSaving, onReset }: AgentEditorHeaderProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  return (
    <header className="px-8 py-6 border-b border-white/5 flex items-center justify-between shrink-0 z-10 backdrop-blur-md">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-2xl bg-surface-container-high border border-white/10 flex items-center justify-center shadow-lg">
          <Bot className="w-6 h-6 text-primary" />
        </div>
        <div className="space-y-0.5">
          {/* Title — index.title */}
          <input
            type="text"
            value={agent.name}
            onChange={e => onUpdate({ name: e.target.value })}
            className="bg-transparent border-none text-2xl font-black tracking-tighter text-white focus:outline-none focus:ring-0 p-0 m-0"
            placeholder="Agent 名称"
          />
          {/* Alias — index.alias */}
          <div className="flex items-center gap-1.5">
            <Hash className="w-3 h-3 text-slate-500" />
            <input
              type="text"
              value={agent.alias}
              onChange={e => onUpdate({ alias: e.target.value })}
              placeholder="别名标识_如_omni_doll"
              className="bg-transparent border-none text-xs text-slate-500 font-mono focus:outline-none focus:ring-0 p-0 w-48"
            />
          </div>
          {/* Summary — index.summary */}
          <input
            type="text"
            value={agent.summary}
            onChange={e => onUpdate({ summary: e.target.value })}
            placeholder="一句话描述该 Agent..."
            className="bg-transparent border-none text-sm text-primary/80 focus:outline-none focus:ring-0 p-0 mt-0.5 w-full min-w-[320px]"
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => setShowDeleteConfirm(true)}
          className="flex items-center justify-center p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-400/10 transition-all border border-transparent hover:border-red-400/20"
          title="删除 Agent"
        >
          <Trash2 className="w-4 h-4" />
        </button>

        <ConfirmDialog
          isOpen={showDeleteConfirm}
          title="删除 Agent"
          message="确定要删除此 Agent 吗？此操作不可撤销。"
          confirmText="删除"
          onConfirm={onDelete}
          onCancel={() => setShowDeleteConfirm(false)}
        />

        <button
          onClick={() => onUpdate({ status: agent.status === 'Active' ? 'Inactive' : 'Active' })}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-bold transition-all ${
            agent.status === 'Active'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : 'bg-white/5 border-white/10 text-slate-400'
          }`}
        >
          <Power className="w-3.5 h-3.5" />
          {agent.status === 'Active' ? '已启用' : '已停用'}
        </button>

        {isDirty && onReset && (
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 text-slate-400 hover:text-slate-200 transition-colors text-sm font-medium"
            title="撤销更改"
          >
            <RotateCcw className="w-4 h-4" />
            <span>重置</span>
          </button>
        )}

        <button
          onClick={onSave}
          disabled={!isDirty || isSaving}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-xl border transition-all ${
            isDirty
              ? 'bg-primary/20 hover:bg-primary/30 text-primary border-primary/30 shadow-[0_0_15px_rgba(197,154,255,0.2)]'
              : 'bg-white/5 text-slate-500 border-white/5 cursor-not-allowed'
          }`}
        >
          {isSaving ? (
            <span className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
          ) : isDirty ? (
            <Save className="w-4 h-4" />
          ) : (
            <Check className="w-4 h-4" />
          )}
          <span className="text-sm font-bold tracking-wide">
            {isSaving ? '保存中...' : isDirty ? '保存' : '已保存'}
          </span>
        </button>
      </div>
    </header>
  );
}
