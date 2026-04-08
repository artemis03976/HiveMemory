import { useState, useEffect, useCallback } from 'react';
import type {
  MemoryAtom,
  MemoryListResponse,
  SearchMode,
  SortOption,
  ViewMode,
} from '@/types/memory';

import { useMemoryViewStore } from '@/stores/memoryViewStore';
import { MOCK_MEMORIES } from '@/constants/memories';

interface UseMemoriesReturn {
  memories: MemoryAtom[];
  loading: boolean;
  error: string | null;
  total: number;

  // Search & Filter
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  searchMode: SearchMode;
  setSearchMode: (mode: SearchMode) => void;

  // Filters
  selectedType: string | null;
  setSelectedType: (type: string | null) => void;
  selectedTags: string[];
  setSelectedTags: (tags: string[]) => void;
  statusFilter: 'all' | 'active' | 'archived';
  setStatusFilter: (status: 'all' | 'active' | 'archived') => void;

  // Sort & View
  sortBy: SortOption;
  setSortBy: (sort: SortOption) => void;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Actions
  refetch: () => Promise<void>;
  deleteMemory: (id: string) => Promise<void>;
  updateMemory: (id: string, patch: Partial<Pick<MemoryAtom, 'title' | 'summary' | 'content' | 'alias' | 'tags'>>) => Promise<void>;
}

export function useMemories(): UseMemoriesReturn {
  const [memories, setMemories] = useState<MemoryAtom[]>([]);
  const [rawMemories, setRawMemories] = useState<MemoryAtom[]>([]); // 缓存原始数据
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  // Search & Filter state from persisted store
  const {
    searchQuery, setSearchQuery,
    searchMode, setSearchMode,
    selectedType, setSelectedType,
    selectedTags, setSelectedTags,
    statusFilter, setStatusFilter,
    sortBy, setSortBy,
    viewMode, setViewMode,
  } = useMemoryViewStore();

  // Fetch memories from API (only called once on mount)
  const fetchMemories = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/v1/memories?limit=100');

      if (!response.ok) {
        throw new Error(`Failed to fetch memories: ${response.statusText}`);
      }

      const data: MemoryListResponse = await response.json();
      setRawMemories(data.memories);
    } catch (err) {
      console.warn('API fetch failed, using mock data:', err);
      // Use mock data when API is unavailable
      setRawMemories(MOCK_MEMORIES);
      setError(null); // Clear error when using mock data
    } finally {
      setLoading(false);
    }
  }, []);

  // 初始加载时获取数据
  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // 客户端过滤和排序 - 每当过滤条件或原始数据变化时重新计算
  useEffect(() => {
    let filtered = [...rawMemories];

    // Apply search filtering
    if (searchQuery.trim()) {
      const query = searchQuery.trim().toLowerCase();

      if (searchMode === 'exact') {
        if (query.startsWith('alias:')) {
          const aliasQuery = query.substring(6);
          filtered = filtered.filter(m =>
            m.alias?.toLowerCase().includes(aliasQuery)
          );
        } else if (query.startsWith('tag:')) {
          const tagQuery = query.substring(4);
          filtered = filtered.filter(m =>
            m.tags.some(tag => tag.toLowerCase().includes(tagQuery))
          );
        } else {
          filtered = filtered.filter(m =>
            m.title.toLowerCase().includes(query) ||
            m.summary.toLowerCase().includes(query)
          );
        }
      } else {
        // Semantic mode (for mock data, use simple text matching)
        filtered = filtered.filter(m =>
          m.title.toLowerCase().includes(query) ||
          m.summary.toLowerCase().includes(query) ||
          m.content.toLowerCase().includes(query) ||
          m.tags.some(tag => tag.toLowerCase().includes(query))
        );
      }
    }

    // Filter by type
    if (selectedType) {
      filtered = filtered.filter(m => m.memory_type === selectedType);
    }

    // Filter by selected tags (AND logic)
    if (selectedTags.length > 0) {
      filtered = filtered.filter(m =>
        selectedTags.every(tag => m.tags.includes(tag))
      );
    }

    // Sort memories
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'created_desc':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'created_asc':
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        case 'accessed':
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        case 'count':
          return b.access_count - a.access_count;
        case 'confidence':
          return b.confidence_score - a.confidence_score;
        default:
          return 0;
      }
    });

    setMemories(filtered);
    setTotal(filtered.length);
  }, [rawMemories, searchQuery, searchMode, selectedType, selectedTags, sortBy]);

  const updateMemory = useCallback(async (
    id: string,
    patch: Partial<Pick<MemoryAtom, 'title' | 'summary' | 'content' | 'alias' | 'tags'>>
  ) => {
    const response = await fetch(`/api/v1/memories/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!response.ok) throw new Error(`Failed to update memory: ${response.statusText}`);
    const updated: MemoryAtom = await response.json();
    setRawMemories(prev => prev.map(m => m.id === id ? updated : m));
  }, []);

  // Delete memory with optimistic update
  const deleteMemory = useCallback(async (id: string) => {
    // Optimistic update
    setMemories(prev => prev.filter(m => m.id !== id));
    setRawMemories(prev => prev.filter(m => m.id !== id));
    setTotal(prev => prev - 1);

    try {
      const response = await fetch(`/api/v1/memories/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Failed to delete memory: ${response.statusText}`);
      }
    } catch (err) {
      console.warn('Delete API failed, keeping optimistic update for mock data:', err);
      // For mock data, we keep the optimistic update (don't rollback)
    }
  }, []);

  return {
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
    selectedTags,
    setSelectedTags,
    statusFilter,
    setStatusFilter,
    sortBy,
    setSortBy,
    viewMode,
    setViewMode,
    refetch: fetchMemories,
    deleteMemory,
    updateMemory,
  };
}
