/**
 * Chat API Type Definitions
 *
 * SSE event types matching backend models from:
 * - src/hivememory/server/models/chat.py
 */

// ========== SSE Event Data Types ==========

export interface ChatTokenEvent {
  content: string;
}

export interface MTPStartEvent {
  verb: string;
  target?: string;
  args?: Record<string, unknown>;
  raw_text?: string;
  iteration: number;
}

export interface MTPResultEvent {
  verb: string;
  target?: string;
  args?: Record<string, unknown>;
  raw_text?: string;
  status: string;
  iteration: number;
}

export interface TopicInfoEvent {
  topic_id: string;
  is_new: boolean;
}

export interface ChatDoneEvent {
  final_text: string;
  mtp_iterations: number;
  total_iterations: number;
  mtp_commands_executed: string[];
}

export interface ChatErrorEvent {
  message: string;
  detail?: string;
}

// ========== SSE Event Union Type ==========

export type SSEEventType = 'token' | 'mtp_start' | 'mtp_result' | 'topic_info' | 'done' | 'error';

export interface SSEEvent {
  event: SSEEventType;
  data: ChatTokenEvent | MTPStartEvent | MTPResultEvent | TopicInfoEvent | ChatDoneEvent | ChatErrorEvent;
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
}

// ========== SSE Callbacks ==========

export interface SSECallbacks {
  onToken: (data: ChatTokenEvent) => void;
  onMTPStart: (data: MTPStartEvent) => void;
  onMTPResult: (data: MTPResultEvent) => void;
  onTopicInfo: (data: TopicInfoEvent) => void;
  onDone: (data: ChatDoneEvent) => void;
  onError: (data: ChatErrorEvent) => void;
  onConnectionError: (error: Error) => void;
}
