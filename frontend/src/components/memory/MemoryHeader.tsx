import { Database, Plus, AlertCircle } from 'lucide-react';

interface MemoryHeaderProps {
  totalMemories: number;
  warnings: number;
  onNewMemory: () => void;
}

export default function MemoryHeader({ totalMemories, warnings, onNewMemory }: MemoryHeaderProps) {
  return (
    <header className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-surface/30 backdrop-blur-md z-10 shrink-0">
      <div>
        <h1 className="text-2xl font-black tracking-tighter text-primary drop-shadow-[0_0_12px_rgba(197,154,255,0.3)] flex items-center gap-3">
          <Database className="w-6 h-6" />
          记忆图书馆
        </h1>
        <p className="text-sm text-slate-400 mt-1 font-medium">系统记忆库管理终端</p>
      </div>

      {/* Garden Stats */}
      <div className="flex items-center gap-6 text-xs font-mono text-slate-400 bg-black/20 px-4 py-2 rounded-xl border border-white/5">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 uppercase tracking-widest text-[12px]">记忆总数</span>
          <span className="text-primary font-bold">{totalMemories.toLocaleString()}</span>
        </div>
        <div className="w-px h-4 bg-white/10" />
        <div className={`flex items-center gap-2 ${warnings > 0 ? 'text-red-400' : 'text-slate-500'}`}>
          <AlertCircle className="w-3.5 h-3.5" />
          <span>{warnings} 冲突警告</span>
        </div>
      </div>

      <button 
        onClick={onNewMemory}
        className="flex items-center gap-2 px-4 py-2 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all shadow-[0_0_15px_rgba(197,154,255,0.2)] hover:shadow-[0_0_25px_rgba(197,154,255,0.4)] group"
      >
        <Plus className="w-4 h-4 transition-transform group-hover:rotate-90 duration-300" />
        <span className="text-sm font-bold tracking-wide">注入记忆</span>
      </button>
    </header>
  );
}