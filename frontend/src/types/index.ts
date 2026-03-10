// MTP Protocol Types
export interface MtpAction {
  id: string;
  type: 'RUN' | 'READ' | 'WRITE' | 'SEARCH';
  command: string;
  params?: Record<string, unknown>;
  status: 'pending' | 'executing' | 'success' | 'error';
  response?: string;
  timestamp: number;
}

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
  timestamp: number;
  mtpActions?: MtpAction[];
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
