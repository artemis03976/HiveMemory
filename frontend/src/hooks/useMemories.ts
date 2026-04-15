import { useState, useEffect, useCallback } from 'react';
import type {
  MemoryAtom,
  MemoryListResponse,
  SearchMode,
  SortOption,
  ViewMode,
} from '@/types/memory';
import { useMemoryViewStore } from '@/stores/memoryViewStore';

// Mock data for development/testing when backend is not available
const mockMemories: MemoryAtom[] = [
  {
    id: 'mock-1',
    title: 'HiveMemory 项目架构设计',
    summary: 'HiveMemory 采用三层架构：感知层（Perception）负责消息接收和话题管理，记忆层（Memory）负责向量存储和检索，生成层（Generation）负责 LLM 调用和响应生成。',
    memory_type: 'FACT',
    tags: ['architecture', 'design', 'hivememory'],
    alias: 'fact_hivememory_architecture',
    content: `# HiveMemory 架构设计

## 三层架构

### 1. 感知层 (Perception Layer)
- 负责消息接收和预处理
- 话题管理和路由
- MTP 协议解析

### 2. 记忆层 (Memory Layer)
- Qdrant 向量数据库存储
- BGE-M3 嵌入模型
- 语义检索和相似度匹配

### 3. 生成层 (Generation Layer)
- LLM 调用（Gateway/Librarian/Worker）
- 上下文注入
- 流式响应生成

## 核心特性
- 自动记忆生成和遗忘机制
- 多话题并行处理
- 记忆置信度评分系统`,
    created_at: '2025-01-15T10:30:00Z',
    updated_at: '2025-01-15T10:30:00Z',
    confidence_score: 0.95,
    vitality_score: 88.5,
    user_id: 'user-001',
    access_count: 12,
  },
  {
    id: 'mock-2',
    title: 'React useEffect 清理函数最佳实践',
    summary: '在 useEffect 中返回清理函数用于取消订阅、清除定时器、中止请求等，避免内存泄漏。清理函数在组件卸载或依赖项变化时执行。',
    memory_type: 'CODE_SNIPPET',
    tags: ['react', 'hooks', 'javascript', 'best-practice'],
    alias: 'code_react_useeffect_cleanup',
    content: `# React useEffect 清理函数

## 基本用法

\`\`\`typescript
useEffect(() => {
  // 副作用逻辑
  const subscription = api.subscribe();

  // 返回清理函数
  return () => {
    subscription.unsubscribe();
  };
}, [dependencies]);
\`\`\`

## 常见场景

### 1. 清除定时器
\`\`\`typescript
useEffect(() => {
  const timer = setTimeout(() => {
    console.log('Delayed action');
  }, 1000);

  return () => clearTimeout(timer);
}, []);
\`\`\`

### 2. 取消网络请求
\`\`\`typescript
useEffect(() => {
  const controller = new AbortController();

  fetch('/api/data', { signal: controller.signal })
    .then(res => res.json())
    .then(data => setData(data));

  return () => controller.abort();
}, []);
\`\`\`

### 3. 移除事件监听
\`\`\`typescript
useEffect(() => {
  const handleResize = () => {
    setWindowWidth(window.innerWidth);
  };

  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);
\`\`\``,
    created_at: '2025-01-14T15:20:00Z',
    updated_at: '2025-01-16T09:45:00Z',
    confidence_score: 0.92,
    vitality_score: 75.2,
    user_id: 'user-001',
    access_count: 8,
  },
  {
    id: 'mock-3',
    title: 'Tailwind CSS 毛玻璃效果实现',
    summary: '使用 backdrop-blur 和半透明背景色实现毛玻璃（Glassmorphism）效果。需要配合 border 和 shadow 增强层次感。',
    memory_type: 'CODE_SNIPPET',
    tags: ['css', 'tailwind', 'ui', 'glassmorphism'],
    alias: 'code_tailwind_glassmorphism',
    content: `# Tailwind CSS 毛玻璃效果

## 基础实现

\`\`\`css
.glass-card {
  @apply bg-white/10 backdrop-blur-lg border border-white/20;
  @apply shadow-lg hover:bg-white/20 transition-all;
}
\`\`\`

## 进阶技巧

### 1. 多层次毛玻璃
\`\`\`css
/* 导航栏 - 重度模糊 */
.glass-nav {
  @apply bg-black/40 backdrop-blur-3xl border-r border-white/5;
  box-shadow: 2px 0 30px rgba(0, 0, 0, 0.5);
}

/* 侧边栏 - 中度模糊 */
.glass-sidebar {
  @apply bg-background/30 backdrop-blur-lg border-r border-white/5;
}

/* 卡片 - 轻度模糊 */
.glass-card {
  @apply bg-muted/20 backdrop-blur-lg border border-white/10;
}
\`\`\`

### 2. 输入框增强
\`\`\`css
.glass-input {
  @apply bg-background/50 backdrop-blur-xl border border-white/10;
  @apply focus:ring-1 focus:ring-primary/50 focus:border-primary/50;
  @apply focus:bg-background/70 transition-all;
}
\`\`\``,
    created_at: '2025-01-13T14:10:00Z',
    updated_at: '2025-01-13T14:10:00Z',
    confidence_score: 0.88,
    vitality_score: 62.8,
    user_id: 'user-001',
    access_count: 5,
  },
  {
    id: 'mock-4',
    title: 'Qdrant 向量数据库配置',
    summary: 'Qdrant 是高性能向量数据库，支持密集向量和稀疏向量混合检索。配置包括集合创建、向量维度、距离度量等。',
    memory_type: 'URL_RESOURCE',
    tags: ['qdrant', 'vector-db', 'configuration', 'database'],
    alias: 'resource_qdrant_config',
    content: `# Qdrant 向量数据库配置指南

## 基本配置

### 连接设置
- Host: localhost
- HTTP Port: 6333
- gRPC Port: 6334

### 集合配置
\`\`\`python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="memories",
    vectors_config=VectorParams(
        size=1024,  # BGE-M3 维度
        distance=Distance.COSINE
    )
)
\`\`\`

## 高级特性

### 1. Payload 索引
为常用字段创建索引以加速过滤：
\`\`\`python
client.create_payload_index(
    collection_name="memories",
    field_name="memory_type",
    field_schema="keyword"
)
\`\`\`

### 2. 混合检索
结合密集向量和稀疏向量：
\`\`\`python
results = client.search(
    collection_name="memories",
    query_vector=dense_vector,
    sparse_vector=sparse_vector,
    limit=10
)
\`\`\`

## 参考资源
- 官方文档: https://qdrant.tech/documentation/
- Python SDK: https://github.com/qdrant/qdrant-client`,
    created_at: '2025-01-12T11:00:00Z',
    updated_at: '2025-01-12T11:00:00Z',
    confidence_score: 0.85,
    vitality_score: 55.0,
    user_id: 'user-001',
    access_count: 3,
  },
  {
    id: 'mock-5',
    title: '用户偏好：喜欢简洁的代码风格',
    summary: '用户倾向于编写简洁、可读性强的代码，避免过度工程化。偏好函数式编程风格，使用 TypeScript 严格模式。',
    memory_type: 'USER_PROFILE',
    tags: ['preference', 'coding-style', 'typescript'],
    alias: 'profile_user_coding_style',
    content: `# 用户编码偏好

## 代码风格
- 简洁优于复杂
- 可读性优先
- 避免过度抽象
- 函数式编程风格

## TypeScript 使用
- 启用严格模式
- 明确的类型注解
- 避免 any 类型
- 使用接口定义数据结构

## 命名规范
- 变量：camelCase
- 组件：PascalCase
- 常量：UPPER_SNAKE_CASE
- 文件：kebab-case

## 注释习惯
- 只在必要时添加注释
- 代码应自解释
- 复杂逻辑需要说明`,
    created_at: '2025-01-10T09:00:00Z',
    updated_at: '2025-01-16T10:30:00Z',
    confidence_score: 0.78,
    vitality_score: 45.5,
    user_id: 'user-001',
    access_count: 15,
  },
  {
    id: 'mock-6',
    title: '记忆花园 UI 实现进行中',
    summary: '正在实现记忆花园面板，包括卡片视图、列表视图、搜索过滤、详情模态框等功能。使用 Framer Motion 实现动画效果。',
    memory_type: 'WORK_IN_PROGRESS',
    tags: ['frontend', 'react', 'ui', 'memory-garden'],
    alias: null,
    content: `# 记忆花园 UI 实现

## 已完成
- [x] TypeScript 类型定义
- [x] useMemories API hook
- [x] MemoryCard 组件
- [x] MemoryGrid 布局
- [x] SearchBar 双模搜索
- [x] FilterControls 筛选器
- [x] MemoryDetailModal 详情弹窗

## 进行中
- [ ] 响应式布局优化
- [ ] 加载骨架屏
- [ ] 错误边界处理

## 待实现
- [ ] 编辑记忆功能
- [ ] 固定/锁定功能
- [ ] 手动创建记忆
- [ ] 批量操作
- [ ] 导出功能

## 技术栈
- React 19 + TypeScript
- Framer Motion 动画
- Tailwind CSS 样式
- React Markdown 渲染`,
    created_at: '2025-01-16T08:00:00Z',
    updated_at: '2025-01-16T12:30:00Z',
    confidence_score: 0.42,
    vitality_score: 92.0,
    user_id: 'user-001',
    access_count: 1,
  },
];

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
      setRawMemories(mockMemories);
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
