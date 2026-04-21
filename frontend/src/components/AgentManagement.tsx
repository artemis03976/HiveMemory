import { Bot, Terminal, Zap, Thermometer } from 'lucide-react';
import { useAgentManagement } from '@/hooks/useAgentManagement';
import { useDraft } from '@/hooks/useDraft';
import { AVAILABLE_TOOLS } from '@/constants/agents';
import { AgentSidebar } from './agent/AgentSidebar';
import { AgentEditorHeader } from './agent/AgentEditorHeader';
import { TagsSection } from './agent/TagsSection';
import { PermissionsSection } from './agent/PermissionsSection';
import { ToolsSection } from './agent/ToolsSection';
import type { AgentData } from '@/types/agent';

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
    allowed_mtp_verbs: [],
    allowed_sys_tools: [],
    language: 'zh',
  },
  tools: [],
};

export default function AgentManagement() {
  const {
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
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

              <div className="w-full h-px bg-white/5" />

              {/* System Instructions — payload.content */}
              <section className="space-y-3">
                <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-primary" />
                  System Instructions
                </label>
                <div className="relative group">
                  <textarea
                    value={draft.systemPrompt}
                    onChange={e => updateDraft({ systemPrompt: e.target.value })}
                    rows={8}
                    className="w-full bg-black/20 border border-white/10 rounded-2xl p-4 text-sm text-slate-300 font-mono resize-y focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all leading-relaxed shadow-inner"
                    placeholder="Enter the core system prompt that defines this agent's persona and behavior..."
                  />
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-slate-500 font-mono">{draft.systemPrompt.length} chars</span>
                  </div>
                </div>
              </section>

              <div className="w-full h-px bg-white/5" />

              {/* Model & Temperature — artifacts.agent_config */}
              <section className="grid grid-cols-2 gap-6">
                <div className="space-y-3">
                  <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    Model
                  </label>
                  <input
                    type="text"
                    value={draft.model}
                    onChange={e => {
                      const model = e.target.value;
                      updateDraft({
                        model,
                        config: { ...draft.config, model_name: model },
                      });
                    }}
                    placeholder="e.g. deepseek/deepseek-chat, gpt-4o, claude-sonnet-4-20250514"
                    className="w-full bg-black/20 border border-white/10 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all font-mono"
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
                    <Thermometer className="w-4 h-4 text-primary" />
                    Temperature
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
                    <span className="text-sm text-white font-mono w-8 text-right">
                      {draft.config.temperature.toFixed(1)}
                    </span>
                  </div>
                </div>
              </section>

              <div className="w-full h-px bg-white/5" />

              <PermissionsSection
                config={draft.config}
                onChange={config => updateDraft({ config })}
              />

              <ToolsSection
                selectedTools={draft.tools}
                onToggleTool={(toolId) => {
                  const allIds = AVAILABLE_TOOLS.map(t => t.id);
                  const isAllAllowed = draft.tools.length === 0;
                  
                  if (isAllAllowed) {
                    updateDraft({ tools: allIds.filter(t => t !== toolId) });
                  } else if (draft.tools.includes(toolId)) {
                    updateDraft({ tools: draft.tools.filter(t => t !== toolId) });
                  } else {
                    const next = [...draft.tools, toolId];
                    updateDraft({ tools: next.length === allIds.length ? [] : next });
                  }
                }}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center bg-surface-container-lowest text-slate-500">
          <Bot className="w-16 h-16 mb-4 opacity-20" />
          <p>Select an agent from the list to view details</p>
        </div>
      )}
    </div>
  );
}
