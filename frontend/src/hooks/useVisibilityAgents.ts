import { useEffect, useMemo, useState } from 'react';
import { useAgents } from '@/hooks/useAgents';
import { fetchActorRecords } from '@/services/accessApi';
import { visibilityAgentGroups, type VisibilityAgentGroups } from '@/utils/memoryVisibility';

export interface VisibilityAgents {
  groups: VisibilityAgentGroups;
  /**
   * 两个来源都读取成功。只有这时才能断定某个目标“已不在列表中”：
   * 任一来源读取失败时，列表不全，不在列表中的目标照常显示、不标为失效。
   */
  complete: boolean;
}

/** 记忆可见范围的候选 Agent：当前 Workspace 的访问记录与 Agent Profile。 */
export function useVisibilityAgents(): VisibilityAgents {
  const { agents, loading: agentsLoading, fetchError: agentsError } = useAgents();
  const [actorAgentIds, setActorAgentIds] = useState<(string | null)[]>([]);
  const [actorsLoaded, setActorsLoaded] = useState(false);

  useEffect(() => {
    fetchActorRecords()
      .then((records) => {
        setActorAgentIds(records.map((r) => r.agent_id));
        setActorsLoaded(true);
      })
      .catch((err) => {
        console.error('[MemoryVisibility] failed to fetch access records', err);
      });
  }, []);

  const groups = useMemo(
    () => visibilityAgentGroups(actorAgentIds, agents.map((a) => a.id)),
    [actorAgentIds, agents],
  );
  return { groups, complete: actorsLoaded && !agentsLoading && agentsError === null };
}
