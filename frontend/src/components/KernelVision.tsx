/**
 * KernelVision - L4 panel shell
 *
 * Thin container providing [Terminal] and [Context] tabs.
 * Terminal rendering is fully delegated to the independent KernelTerminal component.
 */

import { useState } from 'react';
import { Brain, Terminal as TerminalIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MemoryAtom } from '@/types';
import { KernelTerminal } from './KernelTerminal';

// ─── Mock data (will be replaced by real context store later) ──

const mockMemories: MemoryAtom[] = [
  {
    id: 'mem1',
    alias: 'fact_project_env',
    summary: 'HiveMemory 项目使用 Python 3.12 和 FastAPI 框架',
    tags: ['python', 'config', 'environment'],
    payload: 'Full memory content...',
    score: 0.95,
  },
  {
    id: 'mem2',
    alias: 'code_mtp_protocol',
    summary: 'MTP 协议的实现细节和使用方法',
    tags: ['mtp', 'protocol', 'code'],
    payload: 'Full memory content...',
    score: 0.88,
  },
];

// ─── Types ────────────────────────────────────────────────────

interface KernelVisionProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'terminal' | 'context';

// ─── Panel Component ──────────────────────────────────────────

export function KernelVision({ isOpen, onClose }: KernelVisionProps) {
  const [activeTab, setActiveTab] = useState<TabType>('terminal');

  return (
    <div className="glass-panel h-full flex flex-col border-l shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <h2 className="text-sm font-semibold text-foreground">Kernel Vision</h2>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-white/10 transition-colors cursor-pointer"
          aria-label="Close panel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/10">
        <TabButton
          active={activeTab === 'terminal'}
          onClick={() => setActiveTab('terminal')}
          icon={TerminalIcon}
          label="Terminal"
        />
        <TabButton
          active={activeTab === 'context'}
          onClick={() => setActiveTab('context')}
          icon={Brain}
          label="Context"
        />
      </div>

      {/* Content — each tab fills remaining space */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'terminal' && <KernelTerminal />}
        {activeTab === 'context' && <ContextTab memories={mockMemories} />}
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ElementType;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex-1 flex items-center justify-center gap-2 px-4 py-2.5',
        'text-sm font-medium transition-colors duration-200 cursor-pointer',
        active
          ? 'text-primary border-b-2 border-primary'
          : 'text-muted-foreground hover:text-foreground'
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );
}

function ContextTab({ memories }: { memories: MemoryAtom[] }) {
  return (
    <div className="p-4 space-y-3 overflow-y-auto custom-scrollbar h-full">
      <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        当前话题参考记忆 (Top {memories.length})
      </h3>

      {memories.map((memory) => (
        <div key={memory.id} className="glass-card p-3 rounded-lg">
          <div className="flex items-start justify-between gap-2 mb-2">
            <code className="text-xs font-mono text-primary">{memory.alias}</code>
            <span className="text-xs text-muted-foreground">
              {(memory.score! * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-foreground mb-2">{memory.summary}</p>
          <div className="flex flex-wrap gap-1">
            {memory.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-xs"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
