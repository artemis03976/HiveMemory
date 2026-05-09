export * from './types/agent';
export * from './types/chat';
export * from './types/kernel';
export * from './types/memory';
export * from './types/mtp';

import type { MTPVerb } from './types/mtp';

export type NavTab = 'chat' | 'database' | 'agents' | 'theme' | 'terminal' | 'settings';

export interface Topic {
  id: string;
  title: string;
  summary?: string;
  activeNow?: boolean;
  timeAgo?: string;
  model: string;
  messageCount?: number;
  lastActive?: number;
  willEvict?: boolean;
  totalTokens?: number;
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

export type SubAgentContentBlock = TextBlock | MtpBlock;

export interface SubAgentBlock {
  kind: 'sub_agent';
  agentId: string;
  task: string;
  status: 'running' | 'completed' | 'error';
  contentBlocks: SubAgentContentBlock[];
  finalText?: string;
}

export type ContentBlock = TextBlock | MtpBlock | SubAgentBlock;

export interface Message {
  id: string;
  role: 'agent' | 'user' | 'assistant' | 'system';
  content: string;
  contentBlocks?: ContentBlock[];
  mtpAction?: MtpAction;
  timestamp?: number;
  isStreaming?: boolean;
  error?: string;
  agent_id?: string;
}