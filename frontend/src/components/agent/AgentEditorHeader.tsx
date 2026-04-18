import { Bot, Power, Save, Hash } from 'lucide-react';
import type { AgentData } from '@/types/agent';

interface AgentEditorHeaderProps {
  agent: AgentData;
  onUpdate: (updates: Partial<AgentData>) => void;
  onSave: () => void;
}

export function AgentEditorHeader({ agent, onUpdate, onSave }: AgentEditorHeaderProps) {
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
            placeholder="Agent Name"
          />
          {/* Alias — index.alias */}
          <div className="flex items-center gap-1.5">
            <Hash className="w-3 h-3 text-slate-500" />
            <input
              type="text"
              value={agent.alias}
              onChange={e => onUpdate({ alias: e.target.value })}
              placeholder="alias_identifier"
              className="bg-transparent border-none text-xs text-slate-500 font-mono focus:outline-none focus:ring-0 p-0 w-48"
            />
          </div>
          {/* Summary — index.summary */}
          <input
            type="text"
            value={agent.summary}
            onChange={e => onUpdate({ summary: e.target.value })}
            placeholder="One-line description of this agent..."
            className="bg-transparent border-none text-sm text-primary/80 focus:outline-none focus:ring-0 p-0 mt-0.5 w-full min-w-[320px]"
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => onUpdate({ status: agent.status === 'Active' ? 'Inactive' : 'Active' })}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-bold transition-all ${
            agent.status === 'Active'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : 'bg-white/5 border-white/10 text-slate-400'
          }`}
        >
          <Power className="w-3.5 h-3.5" />
          {agent.status}
        </button>

        <button
          onClick={onSave}
          className="flex items-center gap-2 px-4 py-1.5 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all shadow-[0_0_15px_rgba(197,154,255,0.2)]"
        >
          <Save className="w-4 h-4" />
          <span className="text-sm font-bold tracking-wide">Save</span>
        </button>
      </div>
    </header>
  );
}
