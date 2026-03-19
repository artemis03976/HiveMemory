// MTP Protocol Types
export interface MtpAction {
  id: string;
  type: 'RUN' | 'READ' | 'WRITE' | 'SEARCH' | 'UPDATE' | 'UNKNOWN';
  command: string;
  params?: Record<string, unknown>;
  status: 'pending' | 'executing' | 'success' | 'error';
  response?: string;
  timestamp: number;
}

// ========== Content Block (有序内容块) ==========

export interface TextBlock {
  kind: 'text';
  text: string;
}

export interface MtpBlock {
  kind: 'mtp';
  action: MtpAction;
}

export type ContentBlock = TextBlock | MtpBlock;

// Topic/MMU Types
export interface Topic {
  id: string;
  title: string;
  summary: string;
  status: 'active' | 'dormant' | 'swapped';
  lastActive: number;
  messageCount: number;
}

// Message Types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  contentBlocks: ContentBlock[];
  timestamp: number;
  isStreaming?: boolean;
  error?: string;
}

// Memory Atom Types
export interface MemoryAtom {
  id: string;
  alias: string;
  summary: string;
  tags: string[];
  payload: string;
  score?: number;
}

// System Event Types
export interface SystemEvent {
  id: string;
  type: 'routing' | 'mtp_parse' | 'execution' | 'memory_write';
  message: string;
  timestamp: number;
  level: 'info' | 'warning' | 'error';
}

// Agent Config Types
export interface AgentConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

// Re-export chat types
export * from './chat';
