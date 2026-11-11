import { useEffect, useState } from 'react';
import { Bot, Terminal, Zap, Thermometer, ChevronDown } from 'lucide-react';
import { useAgentManagement } from '@/hooks/useAgentManagement';
import { useDraft } from '@/hooks/useDraft';
import { AgentSidebar } from './agent/AgentSidebar';
import { AgentEditorHeader } from './agent/AgentEditorHeader';
import { TagsSection } from './agent/TagsSection';
import type { AgentData } from '@/types/agent';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

const DEFAULT_AGENT: AgentData = {
  id: '',
  alias: '',
  name: '',
  summary: '',
  tags: [],
  systemPrompt: '',
  model: 'default',
  status: 'Inactive',
  config: {
    model_name: 'default',
    temperature: 0.7,
    language: 'zh',
  },
};

export default function AgentManagement() {
  const {
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    loadError,
    createAgent,
    saveAgent,
    deleteAgent,
  } = useAgentManagement();

  const { draft, isDirty, isSaving, updateDraft, save, reset } = useDraft({
    initialData: selectedAgent || DEFAULT_AGENT,
    onSave: async (draftData) => {
      await saveAgent(draftData);
    },
  });
  const [models, setModels] = useState<RegisteredModel[]>([]);
  useEffect(() => {
    fetchModels().then(setModels).catch(() => setModels([]));
  }, []);
  // 读不到注册表时不判断“不在注册表中”，免得把可用的模型误标为无效
  const modelUnregistered =
    models.length > 0 && draft.model !== 'default' && !models.some((m) => m.id === draft.model);

  return (
    <div className="flex flex-1 h-full bg-background overflow-hidden">
      <AgentSidebar
        agents={filteredAgents}
        selectedId={selectedId}
        searchQuery={searchQuery}
        onSelect={setSelectedId}
        onSearch={setSearchQuery}
        onCreate={createAgent}
      />

      {selectedAgent ? (
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-surface-container-lowest relative">
          <div className="absolute inset-0 bg-linear-to-br from-primary/5 via-transparent to-transparent pointer-events-none opacity-50" />

          <AgentEditorHeader
            agent={draft}
            onUpdate={updateDraft}
            onSave={save}
            onDelete={() => deleteAgent(draft.id)}
            isDirty={isDirty}
            isSaving={isSaving}
            onReset={reset}
          />

          <div className="flex-1 overflow-y-auto p-8 z-10 scrollbar-hide">
            <div className="max-w-4xl mx-auto space-y-8 pb-12">
              {/* Tags — index.tags */}
              <TagsSection
                tags={draft.tags}
                onChange={tags => updateDraft({ tags })}
              />

              <div className="w-full h-px bg-ink/5" />

              {/* System Instructions — payload.content */}
              <section className="space-y-3">
                <label className="text-sm font-bold text-fg flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-primary" />
                  系统指令
                </label>
                <div className="relative group">
                  <textarea
                    value={draft.systemPrompt}
                    onChange={e => updateDraft({ systemPrompt: e.target.value })}
                    rows={8}
                    className="w-full bg-field border border-line rounded-2xl p-4 text-sm text-fg-2 font-mono resize-y focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all leading-relaxed shadow-inner"
                    placeholder="输入定义该 Agent 人设与行为的核心系统提示词..."
                  />
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-fg-3 font-mono">{draft.systemPrompt.length} 字符</span>
                  </div>
                </div>
              </section>

              <div className="w-full h-px bg-ink/5" />

              {/* 模型与温度 — payload.agent_config */}
              <section className="grid grid-cols-2 gap-6">
                <div className="space-y-3">
                  <label className="text-sm font-bold text-fg flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    模型
                  </label>
                  <div className="relative">
                    <select
                      value={draft.model}
                      onChange={e => {
                        const model = e.target.value;
                        updateDraft({
                          model,
                          config: { ...draft.config, model_name: model },
                        });
                      }}
                      className={`w-full appearance-none rounded-xl border bg-field py-3 pl-4 pr-10 font-mono text-sm text-fg transition-all focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/50 ${
                        modelUnregistered ? 'border-magic-metal' : 'border-line'
                      }`}
                    >
                      <option value="default">模型注册表的默认模型</option>
                      {draft.model !== 'default' && !models.some((m) => m.id === draft.model) && (
                        <option value={draft.model}>{modelUnregistered ? `${draft.model}（不在注册表中）` : draft.model}</option>
                      )}
                      {models.map((m) => (
                        <option key={m.id} value={m.id}>{m.display_name ? `${m.id} · ${m.display_name}` : m.id}</option>
                      ))}
                    </select>
                    <ChevronDown className="pointer-events-none absolute right-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-3" aria-hidden="true" />
                  </div>
                  <p className={`text-xs ${modelUnregistered ? 'text-magic-metal' : 'text-fg-3'}`}>
                    {modelUnregistered
                      ? `当前值 ${draft.model} 不在模型注册表中，对话时会解析失败；请选择一个已登记的模型。`
                      : '从模型注册表中选择；要添加模型，到“系统设置 · 模型注册表”。'}
                  </p>
                </div>
                <div className="space-y-3">
                  <label className="text-sm font-bold text-fg flex items-center gap-2">
                    <Thermometer className="w-4 h-4 text-primary" />
                    模型温度
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={draft.config.temperature}
                      onChange={e => {
                        const temperature = parseFloat(e.target.value);
                        updateDraft({
                          config: { ...draft.config, temperature },
                        });
                      }}
                      className="flex-1 accent-primary"
                    />
                    <span className="text-sm text-fg font-mono w-8 text-right">
                      {draft.config.temperature.toFixed(1)}
                    </span>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center bg-surface-container-lowest text-fg-3">
          <Bot className="w-16 h-16 mb-4 opacity-20" />
          <p>{loadError ?? '从列表中选择一个 Agent 以查看详情'}</p>
        </div>
      )}
    </div>
  );
}
