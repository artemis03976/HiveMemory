import { FileCode, CheckCircle2, Circle } from 'lucide-react';
import { AVAILABLE_TOOLS } from '@/constants/agents';

interface ToolsSectionProps {
  selectedTools: string[];
  onToggleTool: (toolId: string) => void;
}

export function ToolsSection({ selectedTools, onToggleTool }: ToolsSectionProps) {
  return (
    <section className="space-y-4">
      <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
        <FileCode className="w-4 h-4 text-primary" />
        Available Tools
      </label>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {AVAILABLE_TOOLS.map(tool => {
          const isSelected = selectedTools.includes(tool.id);
          const ToolIcon = tool.icon;
          return (
            <div
              key={tool.id}
              onClick={() => onToggleTool(tool.id)}
              className={`p-3 border rounded-xl flex items-center gap-3 cursor-pointer transition-all ${
                isSelected
                  ? 'bg-primary/10 border-primary/30 shadow-[inset_0_0_12px_rgba(197,154,255,0.1)]'
                  : 'bg-black/20 border-white/5 hover:border-white/10 hover:bg-white/5'
              }`}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isSelected ? 'bg-primary text-black' : 'bg-white/5 text-slate-400'}`}>
                <ToolIcon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className={`text-sm font-bold truncate ${isSelected ? 'text-primary' : 'text-slate-300'}`}>
                  {tool.label}
                </div>
                <div className="text-[10px] text-slate-500 font-mono truncate">{tool.id}</div>
              </div>
              {isSelected ? (
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
