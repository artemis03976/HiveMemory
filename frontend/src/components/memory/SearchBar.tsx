import { Search, X, Sparkles, Hash } from 'lucide-react';
import type { SearchMode } from '@/types/memory';
import { cn } from '@/lib/utils';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
}

export function SearchBar({ value, onChange, mode, onModeChange }: SearchBarProps) {
  const placeholder =
    mode === 'semantic'
      ? '语义搜索记忆...'
      : '精确搜索 (alias:name 或 tag:python)';

  return (
    <div className="flex items-center gap-2 flex-1 max-w-2xl">
      {/* Search input */}
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-2 glass-input rounded-lg text-sm focus:outline-none"
        />
        {value && (
          <button
            onClick={() => onChange('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 hover:bg-white/10 rounded transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
      </div>

      {/* Mode toggle */}
      <div className="flex items-center gap-1 p-1 bg-muted/20 rounded-lg border border-white/10">
        <button
          onClick={() => onModeChange('semantic')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all',
            mode === 'semantic'
              ? 'bg-primary/20 text-primary'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <Sparkles className="w-3.5 h-3.5" />
          语义
        </button>
        <button
          onClick={() => onModeChange('exact')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all',
            mode === 'exact'
              ? 'bg-primary/20 text-primary'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <Hash className="w-3.5 h-3.5" />
          精确
        </button>
      </div>
    </div>
  );
}
