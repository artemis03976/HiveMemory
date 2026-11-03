import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { SearchMode, SortOption, ViewMode } from '@/types/memory';

interface MemoryViewStore {
  searchQuery: string;
  setSearchQuery: (query: string) => void;

  searchMode: SearchMode;
  setSearchMode: (mode: SearchMode) => void;

  selectedType: string | null;
  setSelectedType: (type: string | null) => void;

  selectedTags: string[];
  setSelectedTags: (tags: string[]) => void;

  sortBy: SortOption;
  setSortBy: (sort: SortOption) => void;

  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
}

export const useMemoryViewStore = create<MemoryViewStore>()(
  devtools(
    persist(
      (set) => ({
        searchQuery: '',
        setSearchQuery: (query) => set({ searchQuery: query }),

        searchMode: 'semantic',
        setSearchMode: (mode) => set({ searchMode: mode }),

        selectedType: null,
        setSelectedType: (type) => set({ selectedType: type }),

        selectedTags: [],
        setSelectedTags: (tags) => set({ selectedTags: tags }),

        sortBy: 'created_desc',
        setSortBy: (sort) => set({ sortBy: sort }),

        viewMode: 'grid',
        setViewMode: (mode) => set({ viewMode: mode }),
      }),
      {
        name: 'memory-view-store',
        version: 1,
        partialize: (state) => ({
          searchMode: state.searchMode,
          selectedType: state.selectedType,
          sortBy: state.sortBy,
          viewMode: state.viewMode,
        }),
      },
    ),
    { name: 'MemoryViewStore' },
  ),
);
