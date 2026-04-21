import { Bot, Plus, Search } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { AgentData } from '@/types/agent';

interface AgentSidebarProps {
  agents: AgentData[];
  selectedId: string;
  searchQuery: string;
  onSelect: (id: string) => void;
  onSearch: (query: string) => void;
  onCreate: () => void;
}

export function AgentSidebar({ agents, selectedId, searchQuery, onSelect, onSearch, onCreate }: AgentSidebarProps) {
  return (
    <div className="w-80 flex flex-col border-r border-white/5 bg-surface-dim/40 backdrop-blur-xl shrink-0">
      <div className="p-6 border-b border-white/5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-black tracking-tighter text-white flex items-center gap-2">
            <Bot className="w-5 h-5 text-primary" />
            Agents
          </h2>
          <button
            onClick={onCreate}
            className="p-1.5 bg-primary/20 hover:bg-primary/30 text-primary rounded-lg transition-colors border border-primary/30"
            title="Create New Agent"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        <div className="relative">
          <input
            type="text"
            placeholder="Search agents..."
            value={searchQuery}
            onChange={e => onSearch(e.target.value)}
            className="w-full bg-black/20 border border-white/10 rounded-xl pl-9 pr-4 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-primary/50 transition-colors"
          />
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-hide">
        <AnimatePresence>
          {agents.map(agent => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              onClick={() => onSelect(agent.id)}
              className={`p-3 rounded-xl cursor-pointer border transition-all ${
                selectedId === agent.id
                  ? 'bg-primary/10 border-primary/30 shadow-[inset_0_0_12px_rgba(197,154,255,0.1)]'
                  : 'bg-surface-container border-white/5 hover:bg-surface-container-high'
              }`}
            >
              <div className="flex justify-between items-start mb-0.5">
                <span className="font-bold text-sm text-slate-200 truncate pr-2">{agent.name}</span>
                <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${agent.status === 'Active' ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-slate-600'}`} />
              </div>
              {agent.alias && (
                <div className="text-[10px] text-slate-500 font-mono mb-0.5">#{agent.alias}</div>
              )}
              <div className="text-xs text-slate-400 truncate mb-2">{agent.summary}</div>
              <div className="flex gap-1 flex-wrap">
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-black/30 text-slate-500 border border-white/5">
                  {agent.model}
                </span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-black/30 text-slate-500 border border-white/5">
                  {agent.tools === null ? 'All' : agent.tools.length} Tools
                </span>
                {agent.tags.length > 0 && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary/60 border border-primary/10">
                    {agent.tags.length} Tags
                  </span>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
