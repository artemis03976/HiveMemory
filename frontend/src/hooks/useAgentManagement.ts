import { useState, useCallback, useMemo, useEffect } from 'react';
import { useAgents, type BackendAgent } from './useAgents';
import { MOCK_AGENT_CONFIGS, AVAILABLE_TOOLS } from '@/constants/agents';
import type { AgentData, AgentProfileConfig, MTPVerb } from '@/types';
import { useToastStore } from '@/stores/toastStore';
import { createAgent, saveAgent } from '@/services/agentApi';

const DEFAULT_CONFIG: AgentProfileConfig = {
  model_name: 'default',
  temperature: 0.7,
  allowed_mtp_verbs: [],
  allowed_sys_tools: [],
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
      allowed_mtp_verbs: (cfg?.allowed_mtp_verbs ?? []) as MTPVerb[],
      allowed_sys_tools: cfg?.allowed_sys_tools ?? [],
      language: cfg?.language ?? 'zh',
    },
    tools: cfg?.allowed_sys_tools ?? [],
  };
}

export function useAgentManagement() {
  const { rawAgents, loading: backendLoading, fetchError } = useAgents();
  const [agents, setAgents] = useState<AgentData[]>(MOCK_AGENT_CONFIGS);
  const [selectedId, setSelectedId] = useState<string>(MOCK_AGENT_CONFIGS[0]?.id ?? '');
  const [searchQuery, setSearchQuery] = useState('');
  const [initialized, setInitialized] = useState(false);
  const addToast = useToastStore(s => s.addToast);

  useEffect(() => {
    if (backendLoading || initialized) return;
    setInitialized(true);
    if (fetchError) {
      // 后端连接失败，使用 mock 数据
      return;
    }
    // 后端正常但无数据，清空列表（显示占位）
    const backendData = rawAgents.map(backendToAgentData);
    setAgents(backendData);
    setSelectedId(backendData[0]?.id ?? '');
  }, [rawAgents, backendLoading, fetchError, initialized]);

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

  const updateAgent = useCallback((updates: Partial<AgentData>) => {
    setAgents(prev => prev.map(a => a.id === selectedId ? { ...a, ...updates } : a));
  }, [selectedId]);

  const createAgent = useCallback(async () => {
    const tempId = `pending_${Date.now()}`;
    const newAgent: AgentData = {
      id: tempId,
      alias: `new_agent_${Date.now()}`,
      name: 'New Agent',
      summary: '',
      tags: [],
      systemPrompt: '',
      model: 'default',
      status: 'Inactive',
      config: { ...DEFAULT_CONFIG },
      tools: [],
    };
    setAgents(prev => [newAgent, ...prev]);
    setSelectedId(tempId);
    addToast('创建中...', 'info');
    try {
      const created = await createAgent(newAgent);
      setAgents(prev => prev.map(a => a.id === tempId ? { ...a, id: created.id } : a));
      setSelectedId(created.id);
      addToast('创建成功', 'success');
    } catch {
      setAgents(prev => prev.filter(a => a.id !== tempId));
      addToast('创建失败', 'error');
    }
  }, [addToast]);

  const saveAgent = useCallback(async () => {
    if (!selectedAgent) return;
    addToast('保存中...', 'info');
    try {
      await saveAgent(selectedAgent);
      addToast('保存成功', 'success');
    } catch {
      addToast('保存失败', 'error');
    }
  }, [selectedAgent, addToast]);

  const toggleTool = useCallback((toolId: string) => {
    const allIds = AVAILABLE_TOOLS.map(t => t.id);
    setAgents(prev => prev.map(a => {
      if (a.id !== selectedId) return a;
      const isAllAllowed = a.tools.length === 0;

      if (isAllAllowed) {
        // 全部允许 → 展开为完整列表后移除被点击的工具
        return { ...a, tools: allIds.filter(t => t !== toolId) };
      }

      const has = a.tools.includes(toolId);
      if (has) {
        // 移除工具
        return { ...a, tools: a.tools.filter(t => t !== toolId) };
      } else {
        // 添加工具；如果全部选中则回到空列表（全部允许）
        const next = [...a.tools, toolId];
        return { ...a, tools: next.length === allIds.length ? [] : next };
      }
    }));
  }, [selectedId]);

  return {
    agents,
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    loading: backendLoading,
    updateAgent,
    createAgent,
    saveAgent,
    toggleTool,
  };
}
