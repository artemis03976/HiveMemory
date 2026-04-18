import { Bot, Terminal, Zap } from 'lucide-react';
import { useAgentManagement } from '@/hooks/useAgentManagement';
import { AgentSidebar } from './agent/AgentSidebar';
import { AgentEditorHeader } from './agent/AgentEditorHeader';
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

          <AgentEditorHeader agent={selectedAgent} onUpdate={updateAgent} onSave={() => {}} />

          <div className="flex-1 overflow-y-auto p-8 z-10 scrollbar-hide">
            <div className="max-w-4xl mx-auto space-y-8 pb-12">
              {/* System Instructions */}
              <section className="space-y-3">
                <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-primary" />
                  System Instructions
                </label>
                <div className="relative group">
                  <textarea
                    value={selectedAgent.systemPrompt}
                    onChange={e => updateAgent({ systemPrompt: e.target.value })}
                    rows={6}
                    className="w-full bg-black/20 border border-white/10 rounded-2xl p-4 text-sm text-slate-300 font-mono resize-y focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all leading-relaxed shadow-inner"
                    placeholder="Enter the core system prompt that commands this agent's behavior..."
                  />
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-slate-500 font-mono">{selectedAgent.systemPrompt.length} chars</span>
                  </div>
                </div>
              </section>

              {/* Model Selection */}
              <section className="grid grid-cols-2 gap-6">
                <div className="space-y-3">
                  <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    Model Selection
                  </label>
                  <select
                    value={selectedAgent.model}
                    onChange={e => updateAgent({ model: e.target.value })}
                    className="w-full bg-black/20 border border-white/10 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all cursor-pointer font-mono appearance-none"
                  >
                    <option value="GPT-4o">GPT-4o</option>
                    <option value="Claude 3.5 Sonnet">Claude 3.5 Sonnet</option>
                    <option value="deepseek/deepseek-chat">deepseek/deepseek-chat</option>
                    <option value="Gemini 1.5 Pro">Gemini 1.5 Pro</option>
                  </select>
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
