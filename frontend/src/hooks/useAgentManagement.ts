import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useAgents, type BackendAgent } from './useAgents';
import { MOCK_AGENT_CONFIGS } from '@/constants/agents';
import type { AgentData, AgentProfileConfig, MTPVerb } from '@/types';
import { useToastStore } from '@/stores/toastStore';
import { createAgent, saveAgent, deleteAgent } from '@/services/agentApi';


const DEFAULT_CONFIG: AgentProfileConfig = {
  model_name: 'default',
  temperature: 0.7,
  allowed_mtp_verbs: null,
  allowed_sys_tools: null,
  language: 'zh',
};

function backendToAgentData(b: BackendAgent): AgentData {
  const cfg = b.agent_config;
  return {
    id: b.id,
    alias: b.alias,
    name: b.title,
    summary: b.summary,
    tags: b.tags ?? [],
    systemPrompt: b.content ?? '',
    model: cfg?.model_name ?? 'default',
    status: 'Active',
    config: {
      model_name: cfg?.model_name ?? 'default',
      temperature: cfg?.temperature ?? 0.7,
      allowed_mtp_verbs: cfg?.allowed_mtp_verbs !== undefined ? (cfg.allowed_mtp_verbs as MTPVerb[] | null) : null,
      allowed_sys_tools: cfg?.allowed_sys_tools ?? null,
      language: cfg?.language ?? 'zh',
    },
    tools: cfg?.allowed_sys_tools ?? null,
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
        // 后端连接失败，fallback 到 mock 数据
        setAgents(MOCK_AGENT_CONFIGS);
        setSelectedId(MOCK_AGENT_CONFIGS[0]?.id ?? '');
        return;
      }
      const backendData = rawAgents.map(backendToAgentData);
      setAgents(backendData);
      setSelectedId(backendData[0]?.id ?? '');
    });
  }, [rawAgents, backendLoading, fetchError]);

  const filteredAgents = useMemo(
    () => agents.filter(a =>
      a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.alias.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()))
    ),
    [agents, searchQuery],
  );

  const selectedAgent = useMemo(
    () => agents.find(a => a.id === selectedId),
    [agents, selectedId],
  );

  const handleCreateAgent = useCallback(() => {
    const tempId = `pending_${Date.now()}`;
    const newAgent: AgentData = {
      id: tempId,
      alias: `new_agent_${Date.now()}`,
      name: 'New Agent',
      summary: '请填写agent简介（至少10个字符）',
      tags: [],
      systemPrompt: '',
      model: 'default',
      status: 'Inactive',
      config: { ...DEFAULT_CONFIG },
      tools: null,
    };
    setAgents(prev => [newAgent, ...prev]);
    setSelectedId(tempId);
    addToast('已创建草稿，请编辑后保存', 'info');
  }, [addToast]);

  const handleSaveAgent = useCallback(async (draftData: AgentData) => {
    addToast('保存中...', 'info');
    try {
      if (draftData.id.startsWith('pending_')) {
        const created = await createAgent(draftData);
        setAgents(prev => prev.map(a => (a.id === draftData.id ? { ...draftData, id: created.id } : a)));
        setSelectedId(created.id);
      } else {
        await saveAgent(draftData);
        setAgents(prev => prev.map(a => (a.id === draftData.id ? draftData : a)));
      }
      addToast('保存成功', 'success');
    } catch (err) {
      addToast('保存失败', 'error');
      throw err;
    }
  }, [addToast]);

  const handleDeleteAgent = useCallback(async (agentId: string) => {
    if (agentId.startsWith('pending_')) {
      // 还没保存到后端的，直接从本地列表移除
      setAgents(prev => prev.filter(a => a.id !== agentId));
      setSelectedId(prev => prev === agentId ? '' : prev);
      return;
    }

    addToast('删除中...', 'info');
    try {
      await deleteAgent(agentId);
      setAgents(prev => prev.filter(a => a.id !== agentId));
      setSelectedId(prev => prev === agentId ? '' : prev);
      addToast('删除成功', 'success');
    } catch {
      addToast('删除失败', 'error');
    }
  }, [addToast]);

  return {
    agents,
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    loading: backendLoading,
    createAgent: handleCreateAgent,
    saveAgent: handleSaveAgent,
    deleteAgent: handleDeleteAgent,
  };
}
