import type { LucideIcon } from 'lucide-react';
import type { MTPVerb } from './mtp';

/** 对应后端 AgentProfileConfig */
export interface AgentProfileConfig {
  model_name: string;
  temperature: number;
  allowed_mtp_verbs: MTPVerb[];
  allowed_sys_tools: string[];
  language: string;
}

export interface AgentTool {
  id: string;
  label: string;
  icon: LucideIcon;
}

export type AgentStatus = 'Active' | 'Inactive';

export interface AgentData {
  id: string;
  name: string;
  role: string;
  systemPrompt: string;
  model: string;
  status: AgentStatus;
  config: AgentProfileConfig;
  tools: string[];
}
