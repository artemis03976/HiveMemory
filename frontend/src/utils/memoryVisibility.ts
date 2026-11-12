import { DEFAULT_AGENT_ID, SYSTEM_AGENT_ID } from '@/constants/identity';
import type { MemoryAccessPolicy } from '@/types/memory';

/** 可见范围的完整说明：它只约束 Agent，并且不是保密。 */
export const VISIBILITY_NOTE =
  '可见范围只限定 Agent 的检索与读取，管理页始终能看到全部记忆。它不是保密：之后的对话里再提到同一件事，仍可能提取出一条所有 Agent 可见的新记忆。';

/** 可见范围的候选 Agent，按来源分两组；同一个 Agent 只出现在一组里。 */
export interface VisibilityAgentGroups {
  /** 访问登记中有具体记录、却没有 Agent Profile 的 Agent，即经插件接入的 harness。 */
  external: string[];
  /** 有 Agent Profile 的对话 Agent，含内置的 omni_doll。 */
  chat: string[];
}

/** 下拉框的取值：空串表示所有 Agent，否则是唯一能检索与读取的 Agent。 */
export function policyTarget(policy: MemoryAccessPolicy): string {
  return policy.visibility === 'PRIVATE' ? policy.target_agent_id : '';
}

export function targetPolicy(target: string): MemoryAccessPolicy {
  return target ? { visibility: 'PRIVATE', target_agent_id: target } : { visibility: 'PUBLIC' };
}

/**
 * 合并访问记录与 Agent Profile 两个来源。
 *
 * 用户级记录没有 agent_id，保留的 system 不是 Agent，二者都不能成为目标；
 * 既有访问记录又有 Profile 的 Agent 归入对话 Agent。
 */
export function visibilityAgentGroups(
  actorAgentIds: (string | null)[],
  chatAgentIds: string[],
): VisibilityAgentGroups {
  const chat = [...new Set(chatAgentIds)];
  const external = [...new Set(actorAgentIds)].filter(
    (id): id is string => id !== null && id !== SYSTEM_AGENT_ID && !chat.includes(id),
  );
  return { external, chat };
}

export function isListedAgent(groups: VisibilityAgentGroups, agentId: string): boolean {
  return groups.external.includes(agentId) || groups.chat.includes(agentId);
}

/** 下拉框与详情中的 Agent 名称：内置的 omni_doll 标明内置。 */
export function visibilityAgentLabel(agentId: string): string {
  return agentId === DEFAULT_AGENT_ID ? `${agentId}（内置）` : agentId;
}
