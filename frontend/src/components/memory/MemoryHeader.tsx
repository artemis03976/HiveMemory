import { Database, Plus, AlertCircle } from 'lucide-react';

interface MemoryHeaderProps {
  totalMemories: number;
  /** 已加载的记忆数；少于总数时说明列表没有取全。 */
  loadedMemories: number;
  warnings: number;
  onNewMemory: () => void;
}

export default function MemoryHeader({ totalMemories, loadedMemories, warnings, onNewMemory }: MemoryHeaderProps) {
  return (
    <header className="flex items-center justify-between px-8 py-6 border-b border-line bg-surface/30 backdrop-blur-md z-10 shrink-0">
      <div>
        <h1 className="text-2xl font-black tracking-tighter text-primary flex items-center gap-3">
          <Database className="w-6 h-6" />
          记忆图书馆
        </h1>
        <p className="text-sm text-fg-3 mt-1 font-medium">系统记忆库管理终端</p>
      </div>

      {/* Garden Stats */}
      <div className="flex items-center gap-6 text-xs font-mono text-fg-3 bg-field px-4 py-2 rounded-xl border border-line">
        <div className="flex items-center gap-2">
          <span className="text-fg-3 text-[12px]">记忆总数</span>
          <span className="text-primary font-bold">{totalMemories.toLocaleString()}</span>
          {loadedMemories < totalMemories && (
            <span className="text-magic-metal">（已加载 {loadedMemories.toLocaleString()}）</span>
          )}
        </div>
        <div className="w-px h-4 bg-ink/10" />
        <div className={`flex items-center gap-2 ${warnings > 0 ? 'text-magic-fire' : 'text-fg-3'}`}>
          <AlertCircle className="w-3.5 h-3.5" />
          <span>{warnings} 冲突警告</span>
        </div>
      </div>

      <button 
        onClick={onNewMemory}
        className="flex items-center gap-2 px-4 py-2 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all group"
      >
        <Plus className="w-4 h-4 transition-transform group-hover:rotate-90 duration-300" />
        <span className="text-sm font-bold tracking-wide">注入记忆</span>
      </button>
    </header>
  );
}