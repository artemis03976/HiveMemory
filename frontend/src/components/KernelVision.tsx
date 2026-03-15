import { useState, useEffect, useRef } from 'react';
import { Brain, Terminal as TerminalIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MemoryAtom, SystemEvent } from '@/types';
import type { LogEntry } from '@/types/kernel';
import { useKernelStore } from '@/stores/kernelStore';

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
  const [activeTab, setActiveTab] = useState<TabType>('terminal'); // Default to terminal

  // Consume store state
  const logs = useKernelStore((state) => state.filteredLogs());
  const connection = useKernelStore((state) => state.connection);
  const filters = useKernelStore((state) => state.filters);
  const ui = useKernelStore((state) => state.ui);

  // Consume store actions
  const setLogLevel = useKernelStore((state) => state.setLogLevel);
  const setSearchText = useKernelStore((state) => state.setSearchText);
  const clearLogs = useKernelStore((state) => state.clearLogs);
  const toggleAutoScroll = useKernelStore((state) => state.toggleAutoScroll);
  const togglePause = useKernelStore((state) => state.togglePause);

  return (
    <div className="glass-panel h-full flex flex-col border-l shadow-2xl">
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
        {activeTab === 'terminal' && (
          <TerminalTab
            logs={logs}
            connection={connection}
            filters={filters}
            ui={ui}
            actions={{
              setLogLevel,
              setSearchText,
              clearLogs,
              toggleAutoScroll,
              togglePause,
            }}
          />
        )}
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

function TerminalTab({
  logs,
  connection,
  filters,
  ui,
  actions,
}: {
  logs: LogEntry[];
  connection: any;
  filters: any;
  ui: any;
  actions: any;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (ui.autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, ui.autoScroll]);

  const statusColors = {
    disconnected: 'text-gray-400',
    connecting: 'text-yellow-400',
    connected: 'text-green-400',
    error: 'text-red-400',
    reconnecting: 'text-orange-400',
  };

  const levelColors = {
    DEBUG: 'text-gray-400',
    INFO: 'text-blue-400',
    WARNING: 'text-yellow-400',
    ERROR: 'text-red-400',
    CRITICAL: 'text-red-600 font-bold',
  };

  return (
    <div className="flex flex-col h-full">
      {/* Connection Status Bar */}
      <div className="p-2 border-b border-white/10 flex items-center justify-between bg-black/20">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              'w-2 h-2 rounded-full',
              statusColors[connection.status as keyof typeof statusColors]
            )}
          />
          <span className="text-xs text-muted-foreground">
            {connection.status}
          </span>
          {connection.error && (
            <span className="text-xs text-red-400">({connection.error})</span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={actions.togglePause}
            className="text-xs px-2 py-1 rounded hover:bg-white/10 transition-colors"
            title={ui.isPaused ? 'Resume log ingestion' : 'Pause log ingestion'}
          >
            {ui.isPaused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <button
            onClick={actions.toggleAutoScroll}
            className="text-xs px-2 py-1 rounded hover:bg-white/10 transition-colors"
            title={ui.autoScroll ? 'Disable auto-scroll' : 'Enable auto-scroll'}
          >
            {ui.autoScroll ? '🔒 Lock' : '🔓 Unlock'}
          </button>
          <button
            onClick={actions.clearLogs}
            className="text-xs px-2 py-1 rounded hover:bg-white/10 transition-colors"
            title="Clear all logs"
          >
            🗑 Clear
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="p-2 border-b border-white/10 flex gap-2 bg-black/10">
        <select
          value={filters.logLevel || ''}
          onChange={(e) =>
            actions.setLogLevel(e.target.value || null)
          }
          className="text-xs bg-black/20 border border-white/10 rounded px-2 py-1 text-foreground"
        >
          <option value="">All Levels</option>
          <option value="DEBUG">Debug</option>
          <option value="INFO">Info</option>
          <option value="WARNING">Warning</option>
          <option value="ERROR">Error</option>
          <option value="CRITICAL">Critical</option>
        </select>

        <input
          type="text"
          placeholder="Search logs..."
          value={filters.searchText}
          onChange={(e) => actions.setSearchText(e.target.value)}
          className="text-xs bg-black/20 border border-white/10 rounded px-2 py-1 flex-1 text-foreground placeholder:text-muted-foreground"
        />
      </div>

      {/* Log List */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-xs p-4 space-y-1 bg-black/20"
      >
        {logs.length === 0 ? (
          <div className="text-muted-foreground text-center py-8">
            {connection.status === 'connected'
              ? 'No logs yet. Waiting for backend activity...'
              : 'Connecting to backend...'}
          </div>
        ) : (
          logs.map((log) => (
            <div
              key={log.id}
              className="flex gap-2 hover:bg-white/5 p-1 rounded group"
            >
              <span className="text-muted-foreground shrink-0">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span
                className={cn(
                  'font-semibold shrink-0',
                  levelColors[log.level as keyof typeof levelColors]
                )}
              >
                [{log.level}]
              </span>
              <span className="text-blue-300 shrink-0">{log.logger}</span>
              <span className="text-foreground flex-1">{log.message}</span>
              {log.exception && (
                <span className="text-red-400 shrink-0" title="Has exception">
                  ⚠
                </span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Stats Footer */}
      <div className="p-2 border-t border-white/10 flex items-center justify-between bg-black/20 text-xs text-muted-foreground">
        <span>
          {logs.length} logs {filters.logLevel || filters.searchText ? '(filtered)' : ''}
        </span>
        {ui.isPaused && (
          <span className="text-yellow-400">⏸ Paused</span>
        )}
      </div>
    </div>
  );
}
