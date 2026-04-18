import { Bot, Code2, CheckCircle, Palette, Globe, Terminal, Database, Shield, FileCode, type LucideIcon } from 'lucide-react';
import type { AgentData, AgentTool } from '@/types/agent';

export interface AgentProfile {
  id: string;
  name: string;
  avatarIcon: LucideIcon;
  description: string;
  colorClass: string;
}

export const MOCK_AGENTS: AgentProfile[] = [
  { id: 'default', name: '默认全能助手人偶', avatarIcon: Bot, description: '通用的智能助手', colorClass: 'text-primary' },
  { id: 'coder', name: 'Coder Doll', avatarIcon: Code2, description: '专精代码编写与重构', colorClass: 'text-blue-400' },
  { id: 'reviewer', name: 'Reviewer Doll', avatarIcon: CheckCircle, description: '严格的代码审查专家', colorClass: 'text-emerald-400' },
  { id: 'designer', name: 'Designer Doll', avatarIcon: Palette, description: '精通UI/UX设计的画师', colorClass: 'text-pink-400' },
];

export const AVAILABLE_TOOLS: AgentTool[] = [
  { id: 'web_search', label: 'Web Search', icon: Globe },
  { id: 'code_interpreter', label: 'Code Interpreter', icon: Terminal },
  { id: 'query_vector_db', label: 'Query Vector DB', icon: Database },
  { id: 'write_file', label: 'Write File', icon: FileCode },
  { id: 'read_file', label: 'Read File', icon: FileCode },
  { id: 'scan_payload', label: 'Scan Payload', icon: Shield },
  { id: 'semantic_search', label: 'Semantic Search', icon: Database },
  { id: 'update_memory_atom', label: 'Update Memory Atom', icon: Database },
];

export const MOCK_AGENT_CONFIGS: AgentData[] = [
  {
    id: 'agent_1',
    name: 'System Librarian',
    role: 'Memory Graph & Vector DB Manager',
    systemPrompt: 'You are the System Librarian. Your primary responsibility is to manage, categorize, and retrieve information from the HiveMemory vector database. Maintain high accuracy and deduce relationships between atoms.',
    model: 'deepseek/deepseek-chat',
    status: 'Active',
    config: {
      model_name: 'deepseek/deepseek-chat',
      temperature: 0.7,
      allowed_mtp_verbs: ['SEARCH', 'READ', 'WRITE', 'UPDATE'],
      allowed_sys_tools: [],
      language: 'zh',
    },
    tools: ['query_vector_db', 'update_memory_atom', 'semantic_search'],
  },
  {
    id: 'agent_2',
    name: 'Security Bot',
    role: 'Auth & Sanity Checker',
    systemPrompt: 'You are the Security Bot. Inspect every payload and code snippet for malicious intent, memory leaks, and unauthorized access patterns.',
    model: 'GPT-4o',
    status: 'Active',
    config: {
      model_name: 'GPT-4o',
      temperature: 0.3,
      allowed_mtp_verbs: ['SEARCH', 'READ', 'RUN'],
      allowed_sys_tools: [],
      language: 'en',
    },
    tools: ['scan_payload', 'read_file'],
  },
  {
    id: 'agent_3',
    name: 'Data Engineer',
    role: 'Data Pipeline Specialist',
    systemPrompt: 'You analyze data logs, refine extraction procedures, and construct ETL pipes.',
    model: 'Claude 3.5 Sonnet',
    status: 'Inactive',
    config: {
      model_name: 'Claude 3.5 Sonnet',
      temperature: 0.7,
      allowed_mtp_verbs: [],
      allowed_sys_tools: [],
      language: 'zh',
    },
    tools: ['web_search', 'code_interpreter', 'write_file'],
  },
];
