export type MemoryGenerationTaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'cancelled'
  | 'failed';

/** 与后端 `MemoryGenerationSource` 一致：主动写入、更新、话题结算与合并。 */
export type MemoryGenerationTaskSource =
  | 'WRITE'
  | 'UPDATE'
  | 'SETTLE'
  | 'MERGE';

export interface MemoryGenerationTask {
  task_id: string;
  topic_id: string;
  label: string;
  source: MemoryGenerationTaskSource;
  pending_alias: string | null;
  status: MemoryGenerationTaskStatus;
  canonical_alias: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  cancel_requested?: boolean;
  cancelled?: boolean;
  reason?: string | null;
}

export interface MemoryTaskListResponse {
  tasks: MemoryGenerationTask[];
}

export type MemoryTaskConnectionStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface MemoryTaskConnectionState {
  status: MemoryTaskConnectionStatus;
  error: string | null;
  lastFetchedAt: number | null;
}
