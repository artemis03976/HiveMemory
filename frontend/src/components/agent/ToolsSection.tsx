import { FileCode, CheckCircle2, Circle } from 'lucide-react';
import { AVAILABLE_TOOLS } from '@/constants/agents';

interface ToolsSectionProps {
  selectedTools: string[];
  onToggleTool: (toolId: string) => void;
}

export function ToolsSection({ selectedTools, onToggleTool }: ToolsSectionProps) {
  const isAllAllowed = selectedTools.length === 0;

  const isToolSelected = (toolId: string) =>
    isAllAllowed || selectedTools.includes(toolId);

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <FileCode className="w-4 h-4 text-primary" />
          Kernel Syscall Tools
        </label>
        {isAllAllowed && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            全部允许
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {AVAILABLE_TOOLS.map(tool => {
          const selected = isToolSelected(tool.id);
          const ToolIcon = tool.icon;
          return (
            <div
              key={tool.id}
              onClick={() => onToggleTool(tool.id)}
              className={`p-3 border rounded-xl flex items-center gap-3 cursor-pointer transition-all ${
                selected
                  ? 'bg-primary/10 border-primary/30 shadow-[inset_0_0_12px_rgba(197,154,255,0.1)]'
                  : 'bg-black/20 border-white/5 hover:border-white/10 hover:bg-white/5'
              }`}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${selected ? 'bg-primary text-black' : 'bg-white/5 text-slate-400'}`}>
                <ToolIcon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className={`text-sm font-bold truncate ${selected ? 'text-primary' : 'text-slate-300'}`}>
                  {tool.label}
                </div>
                <div className="text-[10px] text-slate-500 font-mono truncate">{tool.id}</div>
              </div>
              {selected ? (
                <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
              ) : (
                <Circle className="w-4 h-4 text-white/10 shrink-0" />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
