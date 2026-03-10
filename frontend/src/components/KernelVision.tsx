import { useState } from 'react';
import { Brain, Terminal as TerminalIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MemoryAtom, SystemEvent } from '@/types';

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

const mockEvents: SystemEvent[] = [
  {
    id: 'e1',
    type: 'routing',
    message: 'TheEye routed query to Topic T_05',
    timestamp: Date.now() - 5000,
    level: 'info',
  },
  {
    id: 'e2',
    type: 'mtp_parse',
    message: 'Detected instruction: ⟪ READ | [mem_01] ⟫',
    timestamp: Date.now() - 4000,
    level: 'info',
  },
  {
    id: 'e3',
    type: 'execution',
    message: 'Koakuma executing READ... Success (45ms)',
    timestamp: Date.now() - 3000,
    level: 'info',
  },
];

interface KernelVisionProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'context' | 'terminal';

export function KernelVision({ isOpen, onClose }: KernelVisionProps) {
  const [activeTab, setActiveTab] = useState<TabType>('context');

  if (!isOpen) return null;

  return (
    <div className="glass-panel h-screen flex flex-col border-l">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
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
          active={activeTab === 'context'}
          onClick={() => setActiveTab('context')}
          icon={Brain}
          label="Context"
        />
        <TabButton
          active={activeTab === 'terminal'}
          onClick={() => setActiveTab('terminal')}
          icon={TerminalIcon}
          label="Terminal"
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {activeTab === 'context' && <ContextTab memories={mockMemories} />}
        {activeTab === 'terminal' && <TerminalTab events={mockEvents} />}
      </div>
    </div>
  );
}

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
        'flex-1 flex items-center justify-center gap-2 px-4 py-3',
        'text-sm font-medium transition-colors duration-200',
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
    <div className="p-4 space-y-3">
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

function TerminalTab({ events }: { events: SystemEvent[] }) {
  const levelColors = {
    info: 'text-blue-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
  };

  return (
    <div className="p-4 font-mono text-xs space-y-2 bg-black/20">
      {events.map((event) => (
        <div key={event.id} className="flex gap-2">
          <span className="text-muted-foreground">
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
          <span className={cn('font-semibold', levelColors[event.level])}>
            [{event.type.toUpperCase()}]
          </span>
          <span className="text-foreground">{event.message}</span>
        </div>
      ))}
    </div>
  );
}
