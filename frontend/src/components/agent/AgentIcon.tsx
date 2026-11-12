import { Bot, type LucideProps } from 'lucide-react';
import { AGENT_ICONS } from '@/hooks/useAgents';

/** Agent 的图标：内置人偶各有图标，其余用默认图标。 */
export function AgentIcon({ alias, ...props }: { alias: string } & LucideProps) {
  const Icon = AGENT_ICONS[alias] ?? Bot;
  return <Icon {...props} />;
}
