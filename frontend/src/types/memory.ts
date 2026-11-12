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

/**
 * Agent 读取策略：PUBLIC 对 Workspace 内所有 Agent 可见，PRIVATE 只对 target_agent_id 可见。
 * 管理接口不开放 TEAM，页面也只呈现这两种。策略只约束 Agent 的检索与读取，管理页始终能看到全部记忆。
 */
export type MemoryAccessPolicy =
  | { visibility: 'PUBLIC'; target_agent_id?: null }
  | { visibility: 'PRIVATE'; target_agent_id: string };

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
  access_policy: MemoryAccessPolicy;
  
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
