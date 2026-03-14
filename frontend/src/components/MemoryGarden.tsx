import { useState } from 'react';
import { AlertCircle, Loader2, Flower2 } from 'lucide-react';
import { useMemories } from '@/hooks/useMemories';
import type { MemoryResponse } from '@/types/memory';
import { MemoryGardenHeader } from './memory/MemoryGardenHeader';
import { MemoryGrid } from './memory/MemoryGrid';
import { MemoryList } from './memory/MemoryList';
import { MemoryDetailModal } from './memory/MemoryDetailModal';

export function MemoryGarden() {
  const {
    memories,
    loading,
    error,
    total,
    searchQuery,
    setSearchQuery,
    searchMode,
    setSearchMode,
    selectedType,
    setSelectedType,
    sortBy,
    setSortBy,
    viewMode,
    setViewMode,
    refetch,
    deleteMemory,
  } = useMemories();

  const [selectedMemory, setSelectedMemory] = useState<MemoryResponse | null>(null);

  // Loading state
  if (loading && memories.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
          <p className="text-sm text-muted-foreground">正在加载记忆...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error && memories.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 max-w-md text-center">
          <div className="p-4 bg-destructive/20 rounded-full">
            <AlertCircle className="w-8 h-8 text-destructive" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-2">加载失败</h3>
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <button
              onClick={refetch}
              className="px-4 py-2 bg-primary/20 hover:bg-primary/30 border border-primary/30 rounded-lg text-sm transition-all"
            >
              重试
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Empty state
  if (memories.length === 0 && !loading) {
    return (
      <div className="h-screen flex flex-col">
        <MemoryGardenHeader
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          searchMode={searchMode}
          onSearchModeChange={setSearchMode}
          selectedType={selectedType}
          onTypeChange={setSelectedType}
          sortBy={sortBy}
          onSortChange={setSortBy}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          total={total}
          displayCount={memories.length}
        />
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4 max-w-md text-center">
            <div className="p-4 bg-muted/20 rounded-full">
              <Flower2 className="w-12 h-12 text-muted-foreground" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                {searchQuery || selectedType ? '没有找到匹配的记忆' : '还没有记忆'}
              </h3>
              <p className="text-sm text-muted-foreground">
                {searchQuery || selectedType
                  ? '尝试调整搜索条件或筛选器'
                  : '开始对话后，系统会自动生成记忆'}
              </p>
              {(searchQuery || selectedType) && (
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedType(null);
                  }}
                  className="mt-4 px-4 py-2 bg-primary/20 hover:bg-primary/30 border border-primary/30 rounded-lg text-sm transition-all"
                >
                  清除筛选
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <MemoryGardenHeader
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        searchMode={searchMode}
        onSearchModeChange={setSearchMode}
        selectedType={selectedType}
        onTypeChange={setSelectedType}
        sortBy={sortBy}
        onSortChange={setSortBy}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        total={total}
        displayCount={memories.length}
      />

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {viewMode === 'grid' ? (
          <MemoryGrid memories={memories} onCardClick={setSelectedMemory} />
        ) : (
          <MemoryList memories={memories} onItemClick={setSelectedMemory} />
        )}
      </div>

      {/* Detail Modal */}
      <MemoryDetailModal
        memory={selectedMemory}
        onClose={() => setSelectedMemory(null)}
        onDelete={deleteMemory}
      />
    </div>
  );
}
