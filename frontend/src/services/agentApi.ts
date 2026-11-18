import { apiErrorMessage } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import type { AgentData } from '@/types';
import type { BackendAgent } from '@/hooks/useAgents';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch } from '@/transports/connectionStatus';

/** 编辑器中可改的 Profile 字段；agent_id 只在新建时给出，修改时出现会被后端拒绝。 */
function toFields(agent: AgentData) {
  return {
    title: agent.name,
    summary: agent.summary,
    persona: agent.systemPrompt,
    model: agent.config.model,
    temperature: agent.config.temperature,
    top_p: agent.config.top_p,
    enabled: agent.status === 'Active',
  };
}

export async function createAgent(agent: AgentData): Promise<BackendAgent> {
  const res = await backendFetch(apiUrl('/api/v1/agents'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify({ agent_id: agent.agentId, ...toFields(agent) }),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'Agent 操作失败'));
  return res.json();
}

export async function saveAgent(agent: AgentData): Promise<BackendAgent> {
  const res = await backendFetch(apiUrl(`/api/v1/agents/${encodeURIComponent(agent.agentId)}`), {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(toFields(agent)),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'Agent 操作失败'));
  return res.json();
}

export async function deleteAgent(agentId: string) {
  const res = await backendFetch(apiUrl(`/api/v1/agents/${encodeURIComponent(agentId)}`), {
    method: 'DELETE',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'Agent 操作失败'));
}
