import { Bot, Code2, CheckCircle, Palette, type LucideIcon } from 'lucide-react';

export interface AgentProfile {
  id: string;
  name: string;
  avatarIcon: LucideIcon;
  description: string;
  colorClass: string;
}

export const MOCK_AGENTS: AgentProfile[] = [
  { 
    id: 'default', 
    name: '默认全能助手人偶', 
    avatarIcon: Bot, 
    description: '通用的智能助手',
    colorClass: 'text-primary'
  },
  { 
    id: 'coder', 
    name: 'Coder Doll', 
    avatarIcon: Code2, 
    description: '专精代码编写与重构',
    colorClass: 'text-blue-400'
  },
  { 
    id: 'reviewer', 
    name: 'Reviewer Doll', 
    avatarIcon: CheckCircle, 
    description: '严格的代码审查专家',
    colorClass: 'text-emerald-400'
  },
  { 
    id: 'designer', 
    name: 'Designer Doll', 
    avatarIcon: Palette, 
    description: '精通UI/UX设计的画师',
    colorClass: 'text-pink-400'
  },
];
