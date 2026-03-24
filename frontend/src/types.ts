export * from './types/chat';
export * from './types/kernel';
export * from './types/memory';

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
}

export interface MtpAction {
  id?: string;
  type: 'RUN' | 'READ' | 'WRITE' | 'SEARCH' | 'UPDATE' | 'UNKNOWN';
  command: string;
  params?: Record<string, unknown> | string;
  status: 'pending' | 'executing' | 'success' | 'error';
  response?: string;
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

export type ContentBlock = TextBlock | MtpBlock;

export interface Message {
  id: string;
  role: 'agent' | 'user' | 'assistant' | 'system';
  content: string;
  contentBlocks?: ContentBlock[];
  mtpAction?: MtpAction;
  timestamp?: number;
  isStreaming?: boolean;
  error?: string;
}