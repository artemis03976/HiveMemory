import type { AgentData } from '@/types/agent';

export const MOCK_AGENT_CONFIGS: AgentData[] = [
  {
    agentId: 'system_librarian',
    name: 'System Librarian',
    summary: 'Memory Graph & Vector DB Manager',
    systemPrompt: 'You are the System Librarian. Your primary responsibility is to manage, categorize, and retrieve information from the HiveMemory vector database. Maintain high accuracy and deduce relationships between atoms.',
    model: 'deepseek/deepseek-chat',
    status: 'Active',
    config: {
      model: 'deepseek/deepseek-chat',
      temperature: 0.7,
      top_p: null,
    },
    updatedAt: '2026-01-01T00:00:00Z',
  },
  {
    agentId: 'security_bot',
    name: 'Security Bot',
    summary: 'Auth & Sanity Checker',
    systemPrompt: 'You are the Security Bot. Inspect every payload and code snippet for malicious intent, memory leaks, and unauthorized access patterns.',
    model: 'GPT-4o',
    status: 'Active',
    config: {
      model: 'GPT-4o',
      temperature: 0.3,
      top_p: null,
    },
    updatedAt: '2026-01-01T00:00:00Z',
  },
  {
    agentId: 'data_engineer',
    name: 'Data Engineer',
    summary: 'Data Pipeline Specialist',
    systemPrompt: 'You analyze data logs, refine extraction procedures, and construct ETL pipes.',
    model: 'Claude 3.5 Sonnet',
    status: 'Inactive',
    config: {
      model: 'Claude 3.5 Sonnet',
      temperature: 0.7,
      top_p: null,
    },
    updatedAt: '2026-01-01T00:00:00Z',
  },
];
