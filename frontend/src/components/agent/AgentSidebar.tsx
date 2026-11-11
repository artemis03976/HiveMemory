import { Plus, Search } from 'lucide-react';
import { PageTitle } from '@/components/common/PageHeader';
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
    <div className="w-80 flex flex-col border-r border-line bg-surface-dim/40 backdrop-blur-xl shrink-0">
      <div className="p-6 border-b border-line space-y-4">
        <div className="flex items-center justify-between">
          <PageTitle title="Agent" description="人偶们的名称、模型与系统指令" size="section" />
          <button
            onClick={onCreate}
            className="mt-1 p-2 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all group"
            title="新建 Agent"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        <div className="relative">
          <input
            type="text"
            placeholder="搜索 Agent..."
            value={searchQuery}
            onChange={e => onSearch(e.target.value)}
            className="w-full bg-field border border-line rounded-xl pl-9 pr-4 py-2 text-sm text-fg placeholder-fg-3 focus:outline-none focus:border-primary/50 transition-colors"
          />
          <Search className="w-4 h-4 text-fg-3 absolute left-3 top-2.5" />
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
                  ? 'bg-primary/10 border-primary/30'
                  : 'bg-surface-container border-line hover:bg-surface-container-high'
              }`}
            >
              <div className="flex justify-between items-start mb-0.5">
                <span className="font-bold text-sm text-fg truncate pr-2">{agent.name}</span>
                <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${agent.status === 'Active' ? 'bg-magic-wood' : 'bg-fg-3'}`} />
              </div>
              {agent.alias && (
                <div className="text-xs text-fg-3 font-mono mb-0.5">#{agent.alias}</div>
              )}
              <div className="text-xs text-fg-3 truncate mb-2">{agent.summary}</div>
              <div className="flex gap-1 flex-wrap">
                <span className="text-xs px-1.5 py-0.5 rounded bg-field text-fg-3 border border-line">
                  {agent.model}
                </span>
                {agent.tags.length > 0 && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary/60 border border-primary/10">
                    {agent.tags.length} 标签
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
