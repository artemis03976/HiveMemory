import { identityHeaders } from '@/services/identity';
import type { AgentData } from '@/types';

/** 显式组装编辑器支持的图纸字段，避免回写存量配置中的废弃字段。 */
function toPayload(agent: AgentData) {
  return {
    title: agent.name,
    alias: agent.alias,
    summary: agent.summary,
    content: agent.systemPrompt,
    tags: agent.tags,
    agent_config: {
      model_name: agent.config.model_name,
      temperature: agent.config.temperature,
      language: agent.config.language,
    },
  };
}

export async function createAgent(agent: AgentData) {
  const res = await fetch('/api/v1/agents', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(toPayload(agent)),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function saveAgent(agent: AgentData) {
  const res = await fetch(`/api/v1/memories/${agent.id}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(toPayload(agent)),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function deleteAgent(agentId: string) {
  const res = await fetch(`/api/v1/memories/${agentId}`, {
    method: 'DELETE',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
}
