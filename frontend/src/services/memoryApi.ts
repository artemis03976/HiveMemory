import { apiErrorMessage } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import type { MemoryAccessPolicy, MemoryAtom, MemoryListResponse } from '@/types/memory';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch } from '@/transports/connectionStatus';

export type UpdateMemoryPatch = Partial<Pick<
  MemoryAtom,
  'title' | 'summary' | 'content' | 'alias' | 'tags' | 'access_policy'
>>;

interface MemoryPageQuery {
  limit?: number;
  cursor?: string | null;
  query?: string;
}

/** 取一页记忆：没有 query 时按存储顺序翻页，有 query 时按相关度语义检索、不翻页。 */
export async function fetchMemoryPage(
  { limit = 100, cursor = null, query }: MemoryPageQuery = {},
): Promise<MemoryListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor) params.set('cursor', cursor);
  if (query) params.set('query', query);
  // Memory 管理请求统一携带 x-user-id / x-workspace-id 基础选择
  const response = await backendFetch(apiUrl(`/api/v1/memories?${params}`), {
    headers: identityHeaders(),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to fetch memories'));
  }

  return response.json();
}

/**
 * 跟随 next_cursor 取完整个记忆库。
 *
 * 后端按存储顺序翻页，顺序与时间无关，只取一页会得到与时间无关的子集；
 * 因此必须取全后再由页面排序。total 是库内总数。
 */
export async function fetchAllMemories(): Promise<{ memories: MemoryAtom[]; total: number }> {
  const memories: MemoryAtom[] = [];
  let total = 0;
  let cursor: string | null = null;
  do {
    const page = await fetchMemoryPage({ cursor });
    memories.push(...page.memories);
    total = page.total;
    cursor = page.next_cursor;
  } while (cursor);
  return { memories, total };
}

/** 在整个记忆库中语义检索，结果按相关度排列。 */
export async function searchMemories(query: string, limit = 50): Promise<MemoryAtom[]> {
  const page = await fetchMemoryPage({ query, limit });
  return page.memories;
}

export interface CreateMemoryPayload {
  title: string;
  summary: string;
  content: string;
  memory_type: string;
  tags: string[];
  alias?: string | null;
  access_policy?: MemoryAccessPolicy;
}

export async function createMemory(payload: CreateMemoryPayload): Promise<MemoryAtom> {
  const response = await backendFetch(apiUrl('/api/v1/memories'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to create memory'));
  }
  return response.json();
}

export async function updateMemory(
  id: string,
  patch: UpdateMemoryPatch,
): Promise<MemoryAtom> {
  const response = await backendFetch(apiUrl(`/api/v1/memories/${encodeURIComponent(id)}`), {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify(patch),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to update memory'));
  }

  return response.json();
}

export async function deleteMemory(id: string): Promise<void> {
  const response = await backendFetch(apiUrl(`/api/v1/memories/${encodeURIComponent(id)}`), {
    method: 'DELETE',
    headers: identityHeaders(),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to delete memory'));
  }
}

export interface MemoryFeedbackResult {
  success: boolean;
  id: string;
  positive: boolean;
  previous_vitality: number;
  new_vitality: number;
  previous_confidence: number;
  new_confidence: number;
  event_type: string;
}

export async function recordMemoryFeedback(
  memoryId: string,
  positive: boolean,
): Promise<MemoryFeedbackResult> {
  const response = await backendFetch(apiUrl(`/api/v1/memories/${encodeURIComponent(memoryId)}/feedback`), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...identityHeaders(),
    },
    body: JSON.stringify({ positive, source: 'ui.memory_ref' }),
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, 'Failed to record memory feedback'));
  }

  return response.json();
}
