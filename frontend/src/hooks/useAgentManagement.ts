import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useAgents, type BackendAgent } from './useAgents';
import { MOCK_AGENT_CONFIGS } from '@/constants/agents';
import { isDebugMode } from '@/utils/debugMode';
import type { AgentData, AgentProfileConfig } from '@/types';
import { useToastStore } from '@/stores';
import { createAgent, saveAgent, deleteAgent } from '@/services/agentApi';


const DEFAULT_CONFIG: AgentProfileConfig = {
  model: 'default',
  temperature: 0.7,
  top_p: null,
};

function backendToAgentData(b: BackendAgent): AgentData {
  return {
    agentId: b.agent_id,
    name: b.title,
    summary: b.summary,
    systemPrompt: b.persona,
    model: b.model,
    status: b.enabled ? 'Active' : 'Inactive',
    config: {
      model: b.model,
      temperature: b.temperature ?? 0.7,
      top_p: b.top_p,
    },
    updatedAt: b.updated_at,
  };
}

export function useAgentManagement() {
  const { rawAgents, loading: backendLoading, fetchError } = useAgents();
  const [agents, setAgents] = useState<AgentData[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const initialized = useRef(false);
  const addToast = useToastStore(s => s.addToast);

  useEffect(() => {
    if (backendLoading || initialized.current) return;
    initialized.current = true;
    queueMicrotask(() => {
      if (fetchError) {
        // 只有 debug 模式在读取失败时换用 mock 数据；否则列表为空，由页面提示读取失败。
        if (isDebugMode()) {
          setAgents(MOCK_AGENT_CONFIGS);
          setSelectedId(MOCK_AGENT_CONFIGS[0]?.agentId ?? '');
        }
        return;
      }
      const backendData = rawAgents.map(backendToAgentData);
      setAgents(backendData);
      setSelectedId(backendData[0]?.agentId ?? '');
    });
  }, [rawAgents, backendLoading, fetchError]);

  const filteredAgents = useMemo(
    () => agents.filter(a =>
      a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.agentId.toLowerCase().includes(searchQuery.toLowerCase())
    ),
    [agents, searchQuery],
  );

  const selectedAgent = useMemo(
    () => agents.find(a => a.agentId === selectedId),
    [agents, selectedId],
  );

  const handleCreateAgent = useCallback(() => {
    const agentId = `new_agent_${Date.now()}`;
    const newAgent: AgentData = {
      agentId,
      name: 'New Agent',
      summary: '',
      systemPrompt: '',
      model: 'default',
      status: 'Active',
      config: { ...DEFAULT_CONFIG },
    };
    setAgents(prev => [newAgent, ...prev]);
    setSelectedId(agentId);
    addToast('已创建草稿，请编辑后保存', 'info');
  }, [addToast]);

  /**
   * 保存列表中 `agentId` 这一项的编辑结果，列表项换成后端读回的 Profile。草稿的
   * agent_id 在保存前还能改，所以列表项以编辑前的 agent_id 定位。
   */
  const handleSaveAgent = useCallback(async (agentId: string, draftData: AgentData) => {
    addToast('保存中...', 'info');
    try {
      // 尚未保存的草稿没有保存时间
      const isDraft = draftData.updatedAt === undefined;
      const saved = isDraft ? await createAgent(draftData) : await saveAgent(draftData);
      setAgents(prev => prev.map(a => (a.agentId === agentId ? backendToAgentData(saved) : a)));
      if (isDraft) setSelectedId(saved.agent_id);
      addToast('保存成功', 'success');
    } catch (err) {
      addToast('保存失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
      throw err;
    }
  }, [addToast]);

  const handleDeleteAgent = useCallback(async (agentId: string) => {
    if (agents.find(a => a.agentId === agentId)?.updatedAt === undefined) {
      // 还没保存到后端的草稿，直接从本地列表移除
      setAgents(prev => prev.filter(a => a.agentId !== agentId));
      setSelectedId(prev => prev === agentId ? '' : prev);
      return;
    }

    addToast('删除中...', 'info');
    try {
      await deleteAgent(agentId);
      setAgents(prev => prev.filter(a => a.agentId !== agentId));
      setSelectedId(prev => prev === agentId ? '' : prev);
      addToast('删除成功', 'success');
    } catch {
      addToast('删除失败', 'error');
    }
  }, [agents, addToast]);

  return {
    agents,
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    loading: backendLoading,
    loadError: isDebugMode() ? null : fetchError,
    createAgent: handleCreateAgent,
    saveAgent: handleSaveAgent,
    deleteAgent: handleDeleteAgent,
  };
}
