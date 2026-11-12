export * from './types/agent';
export * from './types/attachment';
export * from './types/chat';
export * from './types/identity';
export * from './types/workspace';
export * from './types/memory';
export * from './types/memoryTask';
export * from './types/mtp';

import type { ChatRequestParams } from './types/chat';
import type { MTPVerb } from './types/mtp';

export type NavTab = 'chat' | 'database' | 'agents' | 'theme' | 'terminal' | 'settings';

/** 短期话题池中的一个话题；没有标题时 `title` 为空字符串。 */
export interface Topic {
  id: string;
  title: string;
  /** 话题在聊什么。 */
  topicSummary?: string;
  /** 话题目前的进展。 */
  stateSummary?: string;
  lastTurn?: { user: string; assistant: string };
  /** 最近一轮使用的模型；尚未运行过时缺省。 */
  model?: string;
  blockCount: number;
  /** 最近一次写入的毫秒时间戳，由后端的写入时间换算。 */
  lastActive: number;
  totalTokens: number;
}

/** 用户为一轮对话指定的话题，随发送请求带上。 */
export interface DesignatedTopic {
  id: string;
  title: string;
}

export interface MtpAction {
  id?: string;
  type: MTPVerb | 'UNKNOWN';
  command: string;
  target?: string;
  params?: Record<string, unknown> | string;
  status: 'pending' | 'executing' | 'success' | 'error';
  response?: string;
  resultMessage?: string;
  stats?: Record<string, unknown>;
  timestamp?: number;
}

export interface TextBlock {
  kind: 'text';
  text: string;
}

export interface MtpBlock {
  kind: 'mtp';
  action: MtpAction;
}

export type InlineBlock = TextBlock | MtpBlock;

export interface SubAgentBlock {
  kind: 'sub_agent';
  agentId: string;
  task: string;
  status: 'running' | 'completed' | 'error';
  contentBlocks: InlineBlock[];
  finalText?: string;
}

export type ContentBlock = InlineBlock | SubAgentBlock;

export interface Message {
  interactionId?: string;
  failureReason?: string | null;
  topicId?: string;
  outcome?: 'completed' | 'cancelled' | 'failed';
  id: string;
  role: 'agent' | 'user' | 'assistant' | 'system';
  content: string;
  contentBlocks?: ContentBlock[];
  mtpAction?: MtpAction;
  timestamp?: number;
  isStreaming?: boolean;
  error?: string;
  agent_id?: string;
  /**
   * 用户消息发出时的请求选项：Agent、附件快照、指定话题、记忆检索开关与模型选择。
   * 重新生成按它原样再发一轮；会话历史不保存这些选项，从历史恢复的消息没有它。
   */
  request?: Omit<ChatRequestParams, 'message' | 'session_id'>;
}
