import { useState, useEffect, useCallback, useMemo } from 'react';
import { useDebounce } from 'use-debounce';
import type {
  MemoryAtom,
  SearchMode,
  SortOption,
  ViewMode,
} from '@/types/memory';

import { useMemoryViewStore } from '@/stores';
import { MOCK_MEMORIES } from '@/constants/memories';
import {
  deleteMemory as deleteMemoryApi,
  fetchAllMemories,
  searchMemories,
  updateMemory as updateMemoryApi,
  type UpdateMemoryPatch,
} from '@/services/memoryApi';

/** 语义检索的状态：idle 表示当前没有进行语义检索。 */
export type SemanticSearchStatus = 'idle' | 'loading' | 'done' | 'error';

interface UseMemoriesReturn {
  memories: MemoryAtom[];
  loading: boolean;
  error: string | null;
  /** 库内记忆总数（不含 Agent Profile）。 */
  total: number;
  /** 已加载到页面的记忆数；取全时与 total 相同。 */
  loadedCount: number;

  // Search & Filter
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  searchMode: SearchMode;
  setSearchMode: (mode: SearchMode) => void;
  semanticStatus: SemanticSearchStatus;
  semanticError: string | null;

  // Filters
  selectedType: string | null;
  setSelectedType: (type: string | null) => void;
  selectedTags: string[];
  setSelectedTags: (tags: string[]) => void;

  // Sort & View
  sortBy: SortOption;
  setSortBy: (sort: SortOption) => void;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Actions
  refetch: () => Promise<void>;
  deleteMemory: (id: string) => Promise<void>;
  updateMemory: (id: string, patch: UpdateMemoryPatch) => Promise<void>;
}

/** 本地过滤：alias:/tag: 前缀精确到字段，其余匹配标题、摘要、正文与标签。 */
function matchesLocalQuery(memory: MemoryAtom, query: string): boolean {
  if (query.startsWith('alias:')) {
    return memory.alias?.toLowerCase().includes(query.substring(6)) ?? false;
  }
  if (query.startsWith('tag:')) {
    const tagQuery = query.substring(4);
    return memory.tags.some(tag => tag.toLowerCase().includes(tagQuery));
  }
  return (
    memory.title.toLowerCase().includes(query) ||
    memory.summary.toLowerCase().includes(query) ||
    memory.content.toLowerCase().includes(query) ||
    memory.tags.some(tag => tag.toLowerCase().includes(query))
  );
}

function compareBy(sortBy: SortOption) {
  return (a: MemoryAtom, b: MemoryAtom) => {
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
  };
}

export function useMemories(): UseMemoriesReturn {
  const [rawMemories, setRawMemories] = useState<MemoryAtom[]>([]); // 缓存全部已加载的记忆
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [semanticResults, setSemanticResults] = useState<MemoryAtom[]>([]);
  const [semanticStatus, setSemanticStatus] = useState<SemanticSearchStatus>('idle');
  const [semanticError, setSemanticError] = useState<string | null>(null);

  // Search & Filter state from persisted store
  const {
    searchQuery, setSearchQuery,
    searchMode, setSearchMode,
    selectedType, setSelectedType,
    selectedTags, setSelectedTags,
    sortBy, setSortBy,
    viewMode, setViewMode,
  } = useMemoryViewStore();

  // 取全整个记忆库：后端一页的顺序与时间无关，只取一页会漏掉最新的记忆
  const fetchMemories = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await fetchAllMemories();
      setRawMemories(data.memories);
      setTotal(data.total);
    } catch (err) {
      console.warn('API fetch failed, using mock data:', err);
      // Use mock data when API is unavailable
      setRawMemories(MOCK_MEMORIES);
      setTotal(MOCK_MEMORIES.length);
      setError(null); // Clear error when using mock data
    } finally {
      setLoading(false);
    }
  }, []);

  // 初始加载时获取数据
  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // 语义检索在停止输入后才请求后端：每次请求都要计算一次 embedding
  const [debouncedQuery] = useDebounce(searchQuery.trim(), 400);
  const semanticActive = searchMode === 'semantic' && debouncedQuery !== '';

  useEffect(() => {
    if (!semanticActive) {
      setSemanticStatus('idle');
      setSemanticResults([]);
      setSemanticError(null);
      return;
    }
    let cancelled = false; // 输入已经变化时丢弃旧请求的结果
    setSemanticStatus('loading');
    setSemanticError(null);
    searchMemories(debouncedQuery)
      .then(results => {
        if (cancelled) return;
        setSemanticResults(results);
        setSemanticStatus('done');
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setSemanticResults([]);
        setSemanticError(err instanceof Error ? err.message : '语义检索失败');
        setSemanticStatus('error');
      });
    return () => { cancelled = true; };
  }, [semanticActive, debouncedQuery]);

  // 语义检索结果保持后端的相关度顺序；其余情况在已加载的全部记忆上本地过滤并排序
  const memories = useMemo(() => {
    let filtered = semanticActive ? [...semanticResults] : [...rawMemories];

    const query = searchQuery.trim().toLowerCase();
    if (!semanticActive && searchMode === 'exact' && query) {
      filtered = filtered.filter(m => matchesLocalQuery(m, query));
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

    if (!semanticActive) {
      filtered.sort(compareBy(sortBy));
    }
    return filtered;
  }, [
    rawMemories, semanticResults, semanticActive, searchQuery, searchMode,
    selectedType, selectedTags, sortBy,
  ]);

  const updateMemory = useCallback(async (
    id: string,
    patch: UpdateMemoryPatch
  ) => {
    const updated = await updateMemoryApi(id, patch);
    setRawMemories(prev => prev.map(m => m.id === id ? updated : m));
    setSemanticResults(prev => prev.map(m => m.id === id ? updated : m));
  }, []);

  // 先从页面移除；后端删除失败时放回原处并把错误交给调用方提示
  const deleteMemory = useCallback(async (id: string) => {
    const removed = rawMemories.find(m => m.id === id);
    setRawMemories(prev => prev.filter(m => m.id !== id));
    setTotal(prev => prev - 1);

    try {
      await deleteMemoryApi(id);
    } catch (err) {
      if (removed) {
        setRawMemories(prev => prev.some(m => m.id === id) ? prev : [...prev, removed]);
      }
      setTotal(prev => prev + 1);
      throw err;
    }
    setSemanticResults(prev => prev.filter(m => m.id !== id));
  }, [rawMemories]);

  return {
    memories,
    loading,
    error,
    total,
    loadedCount: rawMemories.length,
    searchQuery,
    setSearchQuery,
    searchMode,
    setSearchMode,
    semanticStatus,
    semanticError,
    selectedType,
    setSelectedType,
    selectedTags,
    setSelectedTags,
    sortBy,
    setSortBy,
    viewMode,
    setViewMode,
    refetch: fetchMemories,
    deleteMemory,
    updateMemory,
  };
}
