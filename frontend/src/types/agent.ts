import type { LucideIcon } from 'lucide-react';
import type { MTPVerb } from './mtp';

/** 对应后端 AgentProfileConfig */
export interface AgentProfileConfig {
  model_name: string;
  temperature: number;
  allowed_mtp_verbs: MTPVerb[] | null;
  allowed_sys_tools: string[] | null;
  language: string;
}

export interface AgentTool {
  id: string;
  label: string;
  icon: LucideIcon;
}

export type AgentStatus = 'Active' | 'Inactive';

export interface AgentData {
  id: string;          // 后端 UUID
  alias: string;       // index.alias — snake_case 标识符
  name: string;        // index.title — 显示名
  summary: string;     // index.summary — 一句话简介
  tags: string[];      // index.tags — 语义标签
  systemPrompt: string; // payload.content — 人格/系统指令
  model: string;       // config.model_name
  status: AgentStatus;
  config: AgentProfileConfig;
  tools: string[] | null;     // config.allowed_sys_tools，null=全部允许
}
