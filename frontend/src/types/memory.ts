// Memory-related TypeScript interfaces for the Memory Garden panel

export type MemoryType =
  | 'CODE_SNIPPET'
  | 'FACT'
  | 'URL_RESOURCE'
  | 'REFLECTION'
  | 'USER_PROFILE'
  | 'WORK_IN_PROGRESS'
  | 'TOOL';

export type MemoryStatus = 'Active' | 'Archived';

// This acts as the main Memory model, combining new-frontend UI needs with backend schema
export interface MemoryAtom {
  id: string;
  title: string;
  summary: string;
  memory_type: MemoryType;
  tags: string[];
  alias: string | null;
  content: string; // Same as payload
  created_at: string;
  updated_at: string;
  confidence_score: number;
  vitality_score: number;
  user_id: string;
  access_count: number;
  
  // UI Specific extensions from new-frontend
  status?: MemoryStatus;
  isPinned?: boolean;
  sourceAgent?: string;
}

export interface MemoryListResponse {
  memories: MemoryAtom[];
  /** 没有检索词时是库内总数；有检索词时是本次返回的条数。 */
  total: number;
  /** 下一页的游标；为 null 表示已经取完，检索结果始终为 null。 */
  next_cursor: string | null;
}

export type SearchMode = 'semantic' | 'exact';
export type SortOption = 'created_desc' | 'created_asc' | 'accessed' | 'count' | 'confidence';
export type ViewMode = 'grid' | 'list';

export interface MemoryFilters {
  type: string | null;
  tags: string[];
  status: 'all' | 'active' | 'archived';
}

// Memory type display names
export const memoryTypeLabels: Record<MemoryType, string> = {
  CODE_SNIPPET: '代码',
  FACT: '事实',
  URL_RESOURCE: '资源',
  REFLECTION: '反思',
  USER_PROFILE: '用户偏好',
  WORK_IN_PROGRESS: '进行中',
  TOOL: '工具',
};
