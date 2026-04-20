import type { AgentData } from '@/types';

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
      allowed_mtp_verbs: agent.config.allowed_mtp_verbs,
      allowed_sys_tools: agent.tools,
      language: agent.config.language,
    },
  };
}

export async function createAgent(agent: AgentData) {
  const res = await fetch('/api/v1/agents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toPayload(agent)),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function saveAgent(agent: AgentData) {
  const res = await fetch(`/api/v1/memories/${agent.id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toPayload(agent)),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function deleteAgent(agentId: string) {
  const res = await fetch(`/api/v1/memories/${agentId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await res.text());
}
