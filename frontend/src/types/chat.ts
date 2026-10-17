/**
 * Chat API Type Definitions
 *
 * SSE event types matching backend models from:
 * - src/hivememory/server/models/chat.py
 */

import type { MemoryAtom } from './memory';
import type { WorkspaceAssetRef } from './attachment';

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
  process_id: string;
  status: Exclude<ChatRunStatus, 'idle'>;
  reason?: string | null;
}

export interface ChatDoneEvent {
  final_text?: string;
  topic_id?: string;
  mtp_commands_executed?: string[];
  process_id?: string | null;
  status?: Exclude<ChatRunStatus, 'idle' | 'preparing' | 'streaming' | 'cancelling' | 'finalizing'>;
  stopped?: boolean;
  reason?: string | null;
  failure_reason?: string | null;
}

/** 压缩进度仅用于本轮显示，不是回复正文或检查点提交凭据。 */
export interface ContextCompactionEvent {
  status: 'started' | 'completed' | 'abandoned' | 'failed';
  layer: 'reduction' | 'summary';
  tokens_before: number;
  tokens_after: number;
  covered_turns: number;
  summarized_turns?: number;
  summary_status?: 'not_needed' | 'completed' | 'abandoned';
  elapsed_ms?: number;
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

export interface ProcessIdEvent {
  process_id: string;
  session_id: string;
}

export interface CommandResultEvent {
  command_id: string;
  status: 'completed' | 'rejected' | 'failed' | 'requires_confirmation' | 'not_implemented';
  message: string;
  data: Record<string, unknown>;
  client_action?: ({ type: string } & Record<string, unknown>) | null;
  error_code?: string | null;
}

// ========== SSE Event Union Type ==========

export type SSEEventType =
  | 'token' | 'mtp_start' | 'mtp_result' | 'memory_refs' | 'command_result' | 'done' | 'error'
  | 'sub_agent_start' | 'sub_agent_end' | 'process_id' | 'run_status' | 'context_compaction';

export interface SSEEvent {
  event: SSEEventType;
  data:
    | ChatTokenEvent
    | MTPStartEvent
    | MTPResultEvent
    | MemoryRefsEvent
    | CommandResultEvent
    | ChatDoneEvent
    | ChatErrorEvent
    | ChatRunStatusEvent
    | ContextCompactionEvent
    | ProcessIdEvent;
}

// ========== Connection State ==========

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ChatConnectionState {
  status: ConnectionStatus;
  error: string | null;
}

// ========== API Request Types ==========

/** 本轮 Chat 请求携带的单个附件选择引用（asset_ref 为绑定 ref，非本地路径） */
export interface ChatAttachmentSelection {
  asset_ref: WorkspaceAssetRef;
  representation_id?: string;
  revision?: number;
  content_hash?: string;
}

export interface ChatRequestParams {
  message: string;
  agent_id?: string;
  session_id?: string | null;
  enable_memory_retrieval?: boolean;
  generation_options?: ChatGenerationOptions;
  /** 发送时冻结的附件选择快照；空数组与缺省等价 */
  attachments?: ChatAttachmentSelection[];
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
  onMemoryRefs: (data: MemoryRefsEvent) => void;
  onCommandResult: (data: CommandResultEvent) => void;
  onDone: (data: ChatDoneEvent) => void;
  onError: (data: ChatErrorEvent) => void;
  onConnectionError: (error: Error) => void;
  onSubAgentStart: (data: SubAgentStartEvent) => void;
  onSubAgentToken: (data: ChatTokenEvent) => void;
  onSubAgentMTPStart: (data: MTPStartEvent) => void;
  onSubAgentMTPResult: (data: MTPResultEvent) => void;
  onSubAgentEnd: (data: SubAgentEndEvent) => void;
  onProcessId: (data: ProcessIdEvent) => void;
  onRunStatus: (data: ChatRunStatusEvent) => void;
  onContextCompaction: (data: ContextCompactionEvent) => void;
}
