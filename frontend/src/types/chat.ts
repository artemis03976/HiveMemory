/**
 * Chat API Type Definitions
 *
 * SSE event types matching backend models from:
 * - src/hivememory/server/models/chat.py
 */

import type { MemoryAtom } from './memory';

// ========== SSE Event Data Types ==========

export interface StreamNamespace {
  scope?: 'main' | 'sub';
  depth?: number;
  agent_id?: string;
  frame_id?: string;
}

export interface ChatTokenEvent extends StreamNamespace {
  content: string;
}

export interface MTPStartEvent extends StreamNamespace {
  verb: string;
  target?: string;
  args?: Record<string, unknown>;
  raw_text?: string;
  iteration: number;
}

export interface MTPResultEvent extends StreamNamespace {
  verb: string;
  target?: string;
  args?: Record<string, unknown>;
  raw_text?: string;
  status: string;
  iteration: number;
  result_message?: string;
  stats?: Record<string, unknown>;
}

export interface TopicPoolInfo {
  topics: Array<{
    topic_id: string;
    topic_title?: string;
    title?: string;
    topic_summary?: string;
    state_summary?: string;
    block_count?: number;
    last_accessed_at?: number;
    total_tokens?: number;
  }>;
  current_count: number;
}

export interface TopicInfoEvent {
  topic_id: string;
  is_new: boolean;
  pool_topics?: TopicPoolInfo['topics'];
}

export type ChatRunStatus =
  | 'idle'
  | 'preparing'
  | 'streaming'
  | 'cancelling'
  | 'cancelled'
  | 'finalizing'
  | 'completed'
  | 'failed';

export interface ChatRunStatusEvent {
  generation_id: string;
  status: Exclude<ChatRunStatus, 'idle'>;
  reason?: string | null;
}

export interface ChatDoneEvent {
  final_text: string;
  mtp_iterations: number;
  total_iterations: number;
  mtp_commands_executed?: string[];
  generation_id?: string | null;
  status?: Exclude<ChatRunStatus, 'idle' | 'preparing' | 'streaming' | 'cancelling' | 'finalizing'>;
  stopped?: boolean;
  reason?: string | null;
  memory_task_ids?: string[];
  pool_topics?: TopicPoolInfo['topics'];
}

export interface ChatErrorEvent {
  message: string;
  detail?: string;
}

export interface SubAgentStartEvent extends StreamNamespace {
  agent_id: string;
  task: string;
  iteration: number;
}

export interface SubAgentEndEvent extends StreamNamespace {
  status: 'success' | 'error';
  final_text?: string;
  iteration: number;
}

export interface MemoryRefsEvent {
  memories: MemoryAtom[];
}

export interface GenerationIdEvent {
  generation_id: string;
}

// ========== SSE Event Union Type ==========

export type SSEEventType =
  | 'token' | 'mtp_start' | 'mtp_result' | 'topic_info' | 'memory_refs' | 'done' | 'error'
  | 'sub_agent_start' | 'sub_agent_end' | 'generation_id' | 'run_status';

export interface SSEEvent {
  event: SSEEventType;
  data: ChatTokenEvent | MTPStartEvent | MTPResultEvent | TopicInfoEvent | ChatDoneEvent | ChatErrorEvent | ChatRunStatusEvent;
}

// ========== Connection State ==========

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ChatConnectionState {
  status: ConnectionStatus;
  error: string | null;
}

// ========== API Request Types ==========

export interface ChatRequestParams {
  message: string;
  user_id?: string;
  agent_id?: string;
  session_id?: string | null;
  enable_memory_retrieval?: boolean;
  generation_options?: ChatGenerationOptions;
}

export interface ChatGenerationOptions {
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}

// ========== SSE Callbacks ==========

export interface SSECallbacks {
  onToken: (data: ChatTokenEvent) => void;
  onMTPStart: (data: MTPStartEvent) => void;
  onMTPResult: (data: MTPResultEvent) => void;
  onTopicInfo: (data: TopicInfoEvent) => void;
  onMemoryRefs: (data: MemoryRefsEvent) => void;
  onDone: (data: ChatDoneEvent) => void;
  onError: (data: ChatErrorEvent) => void;
  onConnectionError: (error: Error) => void;
  onSubAgentStart: (data: SubAgentStartEvent) => void;
  onSubAgentToken: (data: ChatTokenEvent) => void;
  onSubAgentMTPStart: (data: MTPStartEvent) => void;
  onSubAgentMTPResult: (data: MTPResultEvent) => void;
  onSubAgentEnd: (data: SubAgentEndEvent) => void;
  onGenerationId: (data: GenerationIdEvent) => void;
  onRunStatus: (data: ChatRunStatusEvent) => void;
}
