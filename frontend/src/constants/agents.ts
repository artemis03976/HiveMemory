import { Bot, Code2, CheckCircle, Palette, Globe, Terminal, FileCode, Clock, type LucideIcon } from 'lucide-react';
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

/**
 * 与后端 kernel syscall 对齐的可用工具列表
 * 对应 build_kernel_registry / DEFAULT_KERNEL_TOOLS
 */
export const AVAILABLE_TOOLS: AgentTool[] = [
  { id: 'sys_clock',       label: 'Clock',         icon: Clock },
  { id: 'sys_web_search',  label: 'Web Search',    icon: Globe },
  { id: 'sys_read_file',   label: 'Read File',     icon: FileCode },
  { id: 'sys_write_file',  label: 'Write File',    icon: FileCode },
  { id: 'sys_python_repl', label: 'Python REPL',   icon: Terminal },
];

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
      allowed_mtp_verbs: ['SEARCH', 'READ', 'WRITE', 'UPDATE'],
      allowed_sys_tools: [],
      language: 'zh',
    },
    tools: ['sys_web_search', 'sys_read_file'],
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
      allowed_mtp_verbs: ['SEARCH', 'READ', 'RUN'],
      allowed_sys_tools: [],
      language: 'en',
    },
    tools: ['sys_read_file', 'sys_python_repl'],
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
      allowed_mtp_verbs: null,
      allowed_sys_tools: null,
      language: 'zh',
    },
    tools: ['sys_web_search', 'sys_python_repl', 'sys_write_file'],
  },
];
