import { useState, useEffect } from 'react';
import { Bot, Code2, CheckCircle, Palette, type LucideIcon } from 'lucide-react';
import { DEFAULT_AGENT_ID } from '@/constants/identity';

export interface AgentProfile {
  id: string;       // alias (used as agent_id in chat)
  name: string;     // title from backend
  description: string;
  avatarIcon: LucideIcon;
  colorClass: string;
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
    language?: string;
  } | null;
}

const ICON_MAP: Record<string, { icon: LucideIcon; colorClass: string }> = {
  omni_doll:     { icon: Bot,         colorClass: 'text-primary'     },
  coder_doll:    { icon: Code2,       colorClass: 'text-blue-400'    },
  reviewer_doll: { icon: CheckCircle, colorClass: 'text-emerald-400' },
  designer_doll: { icon: Palette,     colorClass: 'text-pink-400'    },
};

const DEFAULT_ICON = { icon: Bot, colorClass: 'text-slate-400' };

function toAgentProfile(b: BackendAgent): AgentProfile {
  const { icon, colorClass } = ICON_MAP[b.alias] ?? DEFAULT_ICON;
  return { id: b.alias, name: b.title, description: b.summary, avatarIcon: icon, colorClass };
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
  colorClass: 'text-primary',
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
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    fetch('/api/v1/agents')
      .then(r => r.json())
      .then((data: BackendAgent[]) => {
        setRawAgents(data);
        setAgents(toChatAgents(data));
      })
      .catch((err) => {
        console.error('[AgentDebug][useAgents] failed to fetch /api/v1/agents', err);
        setFetchError(true);
      })
      .finally(() => setLoading(false));
  }, []);

  return { agents, rawAgents, loading, fetchError };
}
