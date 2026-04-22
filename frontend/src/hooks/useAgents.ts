import { useState, useEffect } from 'react';
import { Bot, Code2, CheckCircle, Palette, type LucideIcon } from 'lucide-react';

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
    allowed_mtp_verbs?: string[];
    allowed_sys_tools?: string[];
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

export function useAgents() {
  const [agents, setAgents] = useState<AgentProfile[]>([]);
  const [rawAgents, setRawAgents] = useState<BackendAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    fetch('/api/v1/agents')
      .then(r => r.json())
      .then((data: BackendAgent[]) => {
        setRawAgents(data);
        setAgents(data.map(toAgentProfile));
      })
      .catch((err) => {
        console.error('[AgentDebug][useAgents] failed to fetch /api/v1/agents', err);
        setFetchError(true);
      })
      .finally(() => setLoading(false));
  }, []);

  return { agents, rawAgents, loading, fetchError };
}
