import { Plus, AlertCircle } from 'lucide-react';
import PageHeader from '@/components/common/PageHeader';

interface MemoryHeaderProps {
  totalMemories: number;
  /** 已加载的记忆数；少于总数时说明列表没有取全。 */
  loadedMemories: number;
  warnings: number;
  onNewMemory: () => void;
}

export default function MemoryHeader({ totalMemories, loadedMemories, warnings, onNewMemory }: MemoryHeaderProps) {
  return (
    <PageHeader title="记忆图书馆" description="帕秋莉保管的长期记忆，以及对话中尚未结算的短期话题">

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
        className="flex h-9 items-center gap-1.5 rounded-xl bg-brand px-4 text-sm font-semibold text-white shadow-glow transition-transform active:scale-95"
      >
        <Plus className="h-4 w-4" />
        注入记忆
      </button>
    </PageHeader>
  );
}