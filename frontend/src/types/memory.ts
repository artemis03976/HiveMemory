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
  total: number;
}

export type SearchMode = 'semantic' | 'exact';
export type SortOption = 'created_desc' | 'created_asc' | 'accessed' | 'count' | 'confidence';
export type ViewMode = 'grid' | 'list';

export interface MemoryFilters {
  type: string | null;
  tags: string[];
  status: 'all' | 'active' | 'archived';
}

// Memory type color mapping (七曜魔法)
export const memoryTypeColors: Record<MemoryType, { color: string; name: string }> = {
  CODE_SNIPPET: { color: 'hsl(190, 90%, 50%)', name: 'water' },
  FACT: { color: 'hsl(150, 80%, 45%)', name: 'wood' },
  URL_RESOURCE: { color: 'hsl(52, 82%, 73%)', name: 'metal' },
  REFLECTION: { color: 'hsl(340, 80%, 60%)', name: 'fire' },
  USER_PROFILE: { color: 'hsl(260, 80%, 65%)', name: 'moon' },
  WORK_IN_PROGRESS: { color: 'hsl(45, 100%, 60%)', name: 'sun' },
  TOOL: { color: 'hsl(30, 20%, 50%)', name: 'earth' }, // Added for TOOL type compatibility
};

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
