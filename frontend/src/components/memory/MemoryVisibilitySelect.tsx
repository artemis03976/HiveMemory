import { ChevronDown, Globe, Lock } from 'lucide-react';
import type { VisibilityAgents } from '@/hooks/useVisibilityAgents';
import { isListedAgent, visibilityAgentLabel } from '@/utils/memoryVisibility';

interface MemoryVisibilitySelectProps {
  /** 空串表示所有 Agent，否则是唯一能检索与读取的 Agent。 */
  value: string;
  onChange: (target: string) => void;
  agents: VisibilityAgents;
  id?: string;
  ariaLabel?: string;
}

/**
 * 记忆的可见范围：一个下拉框，第一项是所有 Agent，其后按来源分组列出各 Agent 的 agent_id。
 *
 * 选择本身就带出目标，因此没有“只对一个 Agent 可见却没选 Agent”的中间状态。
 * 当前目标不在候选中时（Agent 已删除或改名，或列表没有读全）单独列在最前，
 * 用户不改动时原样保留。
 */
export default function MemoryVisibilitySelect({ value, onChange, agents, id, ariaLabel }: MemoryVisibilitySelectProps) {
  const { groups, complete } = agents;
  const unlisted = value !== '' && !isListedAgent(groups, value);
  const Icon = value ? Lock : Globe;

  return (
    <div className="relative">
      <Icon aria-hidden="true" className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
      <select
        id={id}
        aria-label={ariaLabel}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full cursor-pointer appearance-none rounded-lg border border-line bg-field py-2 pl-8 pr-8 text-sm text-fg transition-all focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/50"
      >
        {unlisted && (
          <option value={value}>
            {value}{complete ? '（已不在列表中）' : ''}
          </option>
        )}
        <option value="">所有 Agent</option>
        {groups.external.length > 0 && (
          <optgroup label="外部接入">
            {groups.external.map((agentId) => (
              <option key={agentId} value={agentId}>{visibilityAgentLabel(agentId)}</option>
            ))}
          </optgroup>
        )}
        {groups.chat.length > 0 && (
          <optgroup label="对话 Agent">
            {groups.chat.map((agentId) => (
              <option key={agentId} value={agentId}>{visibilityAgentLabel(agentId)}</option>
            ))}
          </optgroup>
        )}
      </select>
      <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
    </div>
  );
}
