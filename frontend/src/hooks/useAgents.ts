import { useState, useEffect } from 'react';
import { Bot, Code2, CheckCircle, Palette, type LucideIcon } from 'lucide-react';
import { DEFAULT_AGENT_ID } from '@/constants/identity';
import { ApiError, readApiError } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch, localErrorText } from '@/transports/connectionStatus';

export interface AgentProfile {
  id: string;       // alias (used as agent_id in chat)
  name: string;     // title from backend
  description: string;
  avatarIcon: LucideIcon;
  /** 停用的 Agent 不出现在对话的选择与 @ 提及中，消息里仍能显示它的名称。 */
  enabled: boolean;
}

export interface BackendAgent {
  id: string;
  alias: string;
  title: string;
  summary: string;
  tags: string[];
  content: string;     // payload.content — Agent 的人格/系统指令
  agent_config?: {
    model_name?: string;
    temperature?: number;
    top_p?: number | null;
    language?: string;
    enabled?: boolean;
  } | null;
  created_at: string;
  updated_at: string;
}

/** 内置人偶的图标；颜色只表示状态，Agent 图标一律用主色。 */
export const AGENT_ICONS: Record<string, LucideIcon> = {
  omni_doll: Bot,
  coder_doll: Code2,
  reviewer_doll: CheckCircle,
  designer_doll: Palette,
};

/** Agent 的图标：内置人偶各有图标，其余用默认图标。 */
export function agentIcon(alias: string): LucideIcon {
  return AGENT_ICONS[alias] ?? Bot;
}

function toAgentProfile(b: BackendAgent): AgentProfile {
  return {
    id: b.alias,
    name: b.title,
    description: b.summary,
    avatarIcon: agentIcon(b.alias),
    enabled: b.agent_config?.enabled !== false,
  };
}

/** 消息与选择器中的 Agent：取自 Profile 列表，不在列表中的显示默认图标与 agent_id。 */
export function resolveChatAgent(agents: AgentProfile[], agentId: string | undefined): AgentProfile {
  const found = agents.find((a) => a.id === agentId);
  if (found) return found;
  const id = agentId || DEFAULT_AGENT_ID;
  return { id, name: id, description: '', avatarIcon: Bot, enabled: true };
}

/**
 * 内置默认图纸由后端代码提供，没有源记忆原子，因此不会出现在 Profile 列表接口中；
 * 聊天仍可以选择它，Agent 管理页只展示后端真实存在的图纸。
 */
const BUILTIN_AGENT: AgentProfile = {
  id: DEFAULT_AGENT_ID,
  name: '默认全能助手人偶',
  description: '内置的通用助手，无特定人设',
  avatarIcon: Bot,
  enabled: true,
};

/** 聊天可选的 Agent：内置图纸固定在首位，后端已有同名图纸时以后端为准。 */
export function toChatAgents(backendAgents: BackendAgent[]): AgentProfile[] {
  const agents = backendAgents.map(toAgentProfile);
  return agents.some((a) => a.id === BUILTIN_AGENT.id) ? agents : [BUILTIN_AGENT, ...agents];
}

export function useAgents() {
  const [agents, setAgents] = useState<AgentProfile[]>([BUILTIN_AGENT]);
  const [rawAgents, setRawAgents] = useState<BackendAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    backendFetch(apiUrl('/api/v1/agents'), { headers: identityHeaders() })
      .then(async (r) => {
        if (!r.ok) throw await readApiError(r, 'Agent 列表读取失败');
        return r.json();
      })
      .then((data: BackendAgent[]) => {
        setRawAgents(data);
        setAgents(toChatAgents(data));
      })
      .catch((err) => {
        console.error('[AgentDebug][useAgents] failed to fetch /api/v1/agents', err);
        // 后端拒绝时它给出的文案就是原因。
        setFetchError(localErrorText(err instanceof ApiError ? err.message : 'Agent 列表读取失败'));
      })
      .finally(() => setLoading(false));
  }, []);

  return { agents, rawAgents, loading, fetchError };
}
