import { useState } from 'react';
import { Folder, Settings, Archive, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Topic } from '@/types';

const mockTopics: Topic[] = [
  {
    id: 't1',
    title: 'Frontend Development Discussion',
    summary: 'Discussing UI/UX implementation for HiveMemory...',
    status: 'active',
    lastActive: Date.now(),
    messageCount: 15,
  },
  {
    id: 't2',
    title: 'Backend API Refactoring',
    summary: 'Refactoring the MTP protocol handlers...',
    status: 'dormant',
    lastActive: Date.now() - 3600000,
    messageCount: 8,
  },
];

type TabType = 'topics' | 'config';

export function ContextSidebar() {
  const [activeTab, setActiveTab] = useState<TabType>('topics');

  return (
    <div className="glass-panel h-screen flex flex-col border-r">
      {/* Tabs */}
      <div className="flex border-b border-white/10">
        <TabButton
          active={activeTab === 'topics'}
          onClick={() => setActiveTab('topics')}
          icon={Folder}
          label="Topics"
        />
        <TabButton
          active={activeTab === 'config'}
          onClick={() => setActiveTab('config')}
          icon={Settings}
          label="Config"
        />
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {activeTab === 'topics' && <TopicsTab topics={mockTopics} />}
        {activeTab === 'config' && <ConfigTab />}
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

function TopicsTab({ topics }: { topics: Topic[] }) {
  return (
    <div className="p-3 space-y-2">
      {topics.map((topic) => (
        <TopicCard key={topic.id} topic={topic} />
      ))}
    </div>
  );
}

function TopicCard({ topic }: { topic: Topic }) {
  const [isHovered, setIsHovered] = useState(false);

  const statusColors = {
    active: 'bg-green-500',
    dormant: 'bg-yellow-500',
    swapped: 'bg-gray-500',
  };

  return (
    <div
      className={cn(
        'glass-card p-3 rounded-lg cursor-pointer group',
        'transition-all duration-200',
        topic.status === 'active' && 'ring-1 ring-primary/50'
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <div className={cn('w-2 h-2 rounded-full', statusColors[topic.status])} />
            <h3 className="text-sm font-medium text-foreground truncate">
              {topic.title}
            </h3>
          </div>
          <p className="text-xs text-muted-foreground line-clamp-2">
            {topic.summary}
          </p>
          <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
            <span>{topic.messageCount} messages</span>
            <span>•</span>
            <span>{formatTime(topic.lastActive)}</span>
          </div>
        </div>

        {/* Action buttons on hover */}
        {isHovered && (
          <div className="flex gap-1">
            <button
              className="p-1 rounded hover:bg-white/10 transition-colors"
              aria-label="Archive topic"
            >
              <Archive className="w-3 h-3" />
            </button>
            <button
              className="p-1 rounded hover:bg-red-500/20 text-red-400 transition-colors"
              aria-label="Delete topic"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ConfigTab() {
  return (
    <div className="p-4 space-y-4">
      <div>
        <label htmlFor="model-select" className="text-sm font-medium text-foreground mb-2 block">
          Model
        </label>
        <select id="model-select" name="model" className="w-full glass-input px-3 py-2 rounded-lg text-sm">
          <option>GPT-4o</option>
          <option>Claude 3.5 Sonnet</option>
          <option>Gemini Pro</option>
        </select>
      </div>

      <div>
        <label htmlFor="temperature-slider" className="text-sm font-medium text-foreground mb-2 block">
          Temperature: 0.7
        </label>
        <input
          id="temperature-slider"
          name="temperature"
          type="range"
          min="0"
          max="2"
          step="0.1"
          defaultValue="0.7"
          className="w-full"
        />
      </div>

      <div>
        <label htmlFor="max-tokens" className="text-sm font-medium text-foreground mb-2 block">
          Max Tokens
        </label>
        <input
          id="max-tokens"
          name="maxTokens"
          type="number"
          defaultValue="4096"
          className="w-full glass-input px-3 py-2 rounded-lg text-sm"
        />
      </div>
    </div>
  );
}

function formatTime(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
