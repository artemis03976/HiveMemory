export type MemoryGenerationTaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'cancelled'
  | 'failed';

export type MemoryGenerationTaskSource =
  | 'WRITE'
  | 'UPDATE'
  | 'ARCHIVE'
  | 'MERGE'
  | 'SPLIT';

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
