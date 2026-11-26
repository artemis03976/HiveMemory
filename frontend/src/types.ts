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
  /** 较早内容的摘要：被折叠的那部分交互的摘要，没有折叠时缺省。 */
  foldSummary?: string;
  lastTurn?: { user: string; assistant: string };
  /** 最近一轮使用的模型；尚未运行过时缺省。 */
  model?: string;
  blockCount: number;
  /** 最近一次写入的毫秒时间戳，由后端的写入时间换算。 */
  lastActive: number;
  totalTokens: number;
  /** 是否有还没有沉淀为长期记忆的交互；没有时空闲不会再触发沉淀。 */
  hasUnsettledInteractions: boolean;
  /** 话题的进展记录，按写下的顺序；还没有写过一条时为空。 */
  progressRecord: TopicProgressEntry[];
  /** 进展记录还没有覆盖到的交互轮数。 */
  unrecordedTurnCount: number;
  /** 还没有整理的交互中最近的几轮的节选，按轮次先后；后端截取，可能少于总轮数。 */
  unrecordedTurns: TopicTurnExcerpt[];
}

/** 一轮交互的节选：轮次号（话题中从 1 数起）、用户消息与回复，后端已截到固定长度。 */
export interface TopicTurnExcerpt {
  turnNumber: number;
  user: string;
  assistant: string;
}

/** 进展记录的一条：话题自上一条以来有了什么进展，覆盖话题中相邻的若干轮。 */
export interface TopicProgressEntry {
  text: string;
  /** 覆盖的第一轮与最后一轮在话题中的轮次号，从 1 数起。 */
  fromTurn: number;
  toTurn: number;
  /** 写下这一条的毫秒时间戳，由后端的时间换算。 */
  writtenAt: number;
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
