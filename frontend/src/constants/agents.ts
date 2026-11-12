import type { AgentData } from '@/types/agent';

export const MOCK_AGENT_CONFIGS: AgentData[] = [
  {
    id: 'agent_1',
    alias: 'system_librarian',
    name: 'System Librarian',
    summary: 'Memory Graph & Vector DB Manager',
    tags: ['memory', 'vector-db', 'librarian'],
    systemPrompt: 'You are the System Librarian. Your primary responsibility is to manage, categorize, and retrieve information from the HiveMemory vector database. Maintain high accuracy and deduce relationships between atoms.',
    model: 'deepseek/deepseek-chat',
    status: 'Active',
    config: {
      model_name: 'deepseek/deepseek-chat',
      temperature: 0.7,
      top_p: null,
    },
  },
  {
    id: 'agent_2',
    alias: 'security_bot',
    name: 'Security Bot',
    summary: 'Auth & Sanity Checker',
    tags: ['security', 'audit'],
    systemPrompt: 'You are the Security Bot. Inspect every payload and code snippet for malicious intent, memory leaks, and unauthorized access patterns.',
    model: 'GPT-4o',
    status: 'Active',
    config: {
      model_name: 'GPT-4o',
      temperature: 0.3,
      top_p: null,
    },
  },
  {
    id: 'agent_3',
    alias: 'data_engineer',
    name: 'Data Engineer',
    summary: 'Data Pipeline Specialist',
    tags: ['data', 'etl', 'pipeline'],
    systemPrompt: 'You analyze data logs, refine extraction procedures, and construct ETL pipes.',
    model: 'Claude 3.5 Sonnet',
    status: 'Inactive',
    config: {
      model_name: 'Claude 3.5 Sonnet',
      temperature: 0.7,
      top_p: null,
    },
  },
];
