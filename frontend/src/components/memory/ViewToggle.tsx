import { LayoutGrid, List } from 'lucide-react';
import type { ViewMode } from '@/types/memory';
import { cn } from '@/lib/utils';

interface ViewToggleProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

export function ViewToggle({ viewMode, onViewModeChange }: ViewToggleProps) {
  return (
    <div className="flex items-center gap-1 p-1 bg-muted/20 rounded-lg border border-white/10">
      <button
        onClick={() => onViewModeChange('grid')}
        className={cn(
          'p-2 rounded transition-all',
          viewMode === 'grid'
            ? 'bg-primary/20 text-primary'
            : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
        )}
        title="Grid view"
      >
        <LayoutGrid className="w-4 h-4" />
      </button>
      <button
        onClick={() => onViewModeChange('list')}
        className={cn(
          'p-2 rounded transition-all',
          viewMode === 'list'
            ? 'bg-primary/20 text-primary'
            : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
        )}
        title="List view"
      >
        <List className="w-4 h-4" />
      </button>
    </div>
  );
}
