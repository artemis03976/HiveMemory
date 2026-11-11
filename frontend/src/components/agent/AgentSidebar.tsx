import { Plus, Search } from 'lucide-react';
import type { AgentData } from '@/types/agent';
import { AgentIcon } from './AgentIcon';

interface AgentSidebarProps {
  agents: AgentData[];
  selectedId: string;
  searchQuery: string;
  onSelect: (id: string) => void;
  onSearch: (query: string) => void;
  onCreate: () => void;
}

/** 状态色：木为启用；停用为中性色；尚未保存的草稿为金色。 */
function agentState(agent: AgentData): { label: string; text: string; dot: string } {
  if (agent.id.startsWith('pending_')) return { label: '草稿', text: 'text-magic-metal', dot: 'bg-magic-metal' };
  return agent.status === 'Active'
    ? { label: '启用', text: 'text-magic-wood', dot: 'bg-magic-wood' }
    : { label: '已停用', text: 'text-fg-3', dot: 'bg-fg-3' };
}

/** 左栏：Agent 列表，可搜索与新建。 */
export function AgentSidebar({ agents, selectedId, searchQuery, onSelect, onSearch, onCreate }: AgentSidebarProps) {
  return (
    <aside aria-label="Agent 列表" className="flex w-[300px] shrink-0 flex-col border-r border-line bg-surface-container-low">
      <div className="flex flex-col gap-1 px-4.5 pb-3 pt-5.5">
        <h1 className="flex items-center gap-2.5 font-manrope text-2xl font-extrabold text-fg">
          Agent
          <span aria-hidden="true" className="mt-1 h-[3px] w-8 rounded-full bg-brand" />
        </h1>
        <p className="text-[13px] text-fg-2">人偶们的名称、模型与系统指令</p>
      </div>

      <div className="flex gap-2 px-3.5 pb-3">
        <label className="relative block flex-1">
          <span className="sr-only">搜索 Agent</span>
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-[15px] w-[15px] -translate-y-1/2 text-fg-3" aria-hidden="true" />
          <input
            type="text"
            placeholder="搜索 Agent"
            value={searchQuery}
            onChange={(e) => onSearch(e.target.value)}
            className="h-[34px] w-full rounded-[10px] border border-line-strong bg-field pl-8 pr-2.5 text-[13px] text-fg placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
          />
        </label>
        <button
          onClick={onCreate}
          aria-label="新建 Agent"
          title="新建 Agent"
          className="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-[10px] border border-primary/40 bg-primary/15 text-primary transition-colors hover:bg-primary/20"
        >
          <Plus className="h-4 w-4" strokeWidth={2.4} />
        </button>
      </div>

      <div className="scrollbar-thin flex flex-1 flex-col gap-0.5 overflow-y-auto px-2 pb-3.5">
        {agents.length === 0 && (
          <p className="pt-8 text-center text-[13px] text-fg-3">{searchQuery ? '没有匹配的 Agent' : '还没有 Agent'}</p>
        )}
        {agents.map((agent) => {
          const selected = agent.id === selectedId;
          const disabled = agent.status !== 'Active';
          const state = agentState(agent);
          return (
            <button
              key={agent.id}
              onClick={() => onSelect(agent.id)}
              aria-current={selected ? 'true' : undefined}
              className={`relative flex items-center gap-3 rounded-xl py-2.5 pl-3 pr-2.5 text-left transition-colors ${
                selected ? 'bg-primary/15' : 'hover:bg-surface-container-high'
              }`}
            >
              {selected && <span aria-hidden="true" className="absolute bottom-3 left-0 top-3 w-[3px] rounded-full bg-brand" />}
              <span
                aria-hidden="true"
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-[11px] bg-primary/15 text-primary ${disabled ? 'opacity-50' : ''}`}
              >
                <AgentIcon alias={agent.alias} className="h-[18px] w-[18px]" />
              </span>
              <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className={`truncate text-sm font-semibold ${disabled ? 'text-fg-3' : 'text-fg'}`}>{agent.name || '未命名 Agent'}</span>
                <span className="truncate font-mono text-xs text-fg-3">{agent.alias}</span>
              </span>
              <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap text-xs ${state.text}`}>
                <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${state.dot}`} />
                {state.label}
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
