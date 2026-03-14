import type { SearchMode, SortOption, ViewMode } from '@/types/memory';
import { SearchBar } from './SearchBar';
import { FilterControls, SortControls } from './FilterControls';
import { ViewToggle } from './ViewToggle';

interface MemoryGardenHeaderProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  searchMode: SearchMode;
  onSearchModeChange: (mode: SearchMode) => void;
  selectedType: string | null;
  onTypeChange: (type: string | null) => void;
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  total: number;
  displayCount: number;
}

export function MemoryGardenHeader({
  searchQuery,
  onSearchChange,
  searchMode,
  onSearchModeChange,
  selectedType,
  onTypeChange,
  sortBy,
  onSortChange,
  viewMode,
  onViewModeChange,
  total,
  displayCount,
}: MemoryGardenHeaderProps) {
  return (
    <div className="border-b border-white/10 bg-background/50 backdrop-blur-lg">
      {/* Title and stats */}
      <div className="px-6 py-4 border-b border-white/10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-foreground">记忆花园</h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              显示 {displayCount} / {total} 条记忆
            </p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="px-6 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-4 flex-1">
          <SearchBar
            value={searchQuery}
            onChange={onSearchChange}
            mode={searchMode}
            onModeChange={onSearchModeChange}
          />
          <FilterControls selectedType={selectedType} onTypeChange={onTypeChange} />
          <SortControls sortBy={sortBy} onSortChange={onSortChange} />
        </div>
        <ViewToggle viewMode={viewMode} onViewModeChange={onViewModeChange} />
      </div>
    </div>
  );
}
