import { Bot, Terminal, Zap, Thermometer } from 'lucide-react';
import { useAgentManagement } from '@/hooks/useAgentManagement';
import { AgentSidebar } from './agent/AgentSidebar';
import { AgentEditorHeader } from './agent/AgentEditorHeader';
import { TagsSection } from './agent/TagsSection';
import { PermissionsSection } from './agent/PermissionsSection';
import { ToolsSection } from './agent/ToolsSection';

export default function AgentManagement() {
  const {
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    updateAgent,
    createAgent,
    saveAgent,
    deleteAgent,
    toggleTool,
  } = useAgentManagement();

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
            agent={selectedAgent}
            onUpdate={updateAgent}
            onSave={saveAgent}
            onDelete={() => deleteAgent(selectedAgent.id)}
          />

          <div className="flex-1 overflow-y-auto p-8 z-10 scrollbar-hide">
            <div className="max-w-4xl mx-auto space-y-8 pb-12">
              {/* Tags — index.tags */}
              <TagsSection
                tags={selectedAgent.tags}
                onChange={tags => updateAgent({ tags })}
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
                    value={selectedAgent.systemPrompt}
                    onChange={e => updateAgent({ systemPrompt: e.target.value })}
                    rows={8}
                    className="w-full bg-black/20 border border-white/10 rounded-2xl p-4 text-sm text-slate-300 font-mono resize-y focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all leading-relaxed shadow-inner"
                    placeholder="Enter the core system prompt that defines this agent's persona and behavior..."
                  />
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-slate-500 font-mono">{selectedAgent.systemPrompt.length} chars</span>
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
                    value={selectedAgent.model}
                    onChange={e => {
                      const model = e.target.value;
                      updateAgent({
                        model,
                        config: { ...selectedAgent.config, model_name: model },
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
                      value={selectedAgent.config.temperature}
                      onChange={e => {
                        const temperature = parseFloat(e.target.value);
                        updateAgent({
                          config: { ...selectedAgent.config, temperature },
                        });
                      }}
                      className="flex-1 accent-primary"
                    />
                    <span className="text-sm text-white font-mono w-8 text-right">
                      {selectedAgent.config.temperature.toFixed(1)}
                    </span>
                  </div>
                </div>
              </section>

              <div className="w-full h-px bg-white/5" />

              <PermissionsSection
                config={selectedAgent.config}
                onChange={config => updateAgent({ config })}
              />

              <ToolsSection
                selectedTools={selectedAgent.tools}
                onToggleTool={toggleTool}
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
