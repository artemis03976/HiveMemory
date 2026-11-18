import { useState, useEffect } from 'react';
import { Bot, Code2, CheckCircle, Palette, type LucideIcon } from 'lucide-react';
import { DEFAULT_AGENT_ID } from '@/constants/identity';
import { ApiError, readApiError } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch, localErrorText } from '@/transports/connectionStatus';

export interface AgentProfile {
  id: string;       // agent_id
  name: string;     // title from backend
  description: string;
  avatarIcon: LucideIcon;
  /** 停用的 Agent 不出现在对话的选择与 @ 提及中，消息里仍能显示它的名称。 */
  enabled: boolean;
}

/** Agent 列表接口返回的一份 Profile。 */
export interface BackendAgent {
  agent_id: string;
  title: string;
  summary: string;
  persona: string;     // Agent 的人格/系统指令
  model: string;
  /** 为空时沿用模型定义的默认值。 */
  temperature: number | null;
  top_p: number | null;
  enabled: boolean;
  /** Profile 文件的修改时间。 */
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
export function agentIcon(agentId: string): LucideIcon {
  return AGENT_ICONS[agentId] ?? Bot;
}

/** 没有起名字的 Profile 以 agent_id 显示。 */
function toAgentProfile(b: BackendAgent): AgentProfile {
  return {
    id: b.agent_id,
    name: b.title || b.agent_id,
    description: b.summary,
    avatarIcon: agentIcon(b.agent_id),
    enabled: b.enabled,
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
 * 内置默认图纸由后端代码提供，不对应 Profile 文件，它的 agent_id 也不能用于新建，
 * 因此不会出现在 Profile 列表接口中；聊天仍可以选择它，Agent 管理页只展示 Profile 文件。
 */
const BUILTIN_AGENT: AgentProfile = {
  id: DEFAULT_AGENT_ID,
  name: '默认全能助手人偶',
  description: '内置的通用助手，无特定人设',
  avatarIcon: Bot,
  enabled: true,
};

/** 聊天可选的 Agent：内置图纸固定在首位，其后是后端列出的 Profile。 */
export function toChatAgents(backendAgents: BackendAgent[]): AgentProfile[] {
  return [BUILTIN_AGENT, ...backendAgents.map(toAgentProfile)];
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
