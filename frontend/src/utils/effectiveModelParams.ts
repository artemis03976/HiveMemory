import type { RegisteredModel } from '@/types/model';

/** Profile 中的模型参数；生成参数为 null 时表示沿用模型定义。 */
export interface AgentModelConfig {
  model_name?: string;
  temperature?: number | null;
  top_p?: number | null;
}

export interface EffectiveParam {
  label: string;
  value: string;
  source: '本次会话' | '来自 Agent' | '模型默认' | '未知';
}

/**
 * 不为本次会话单独设置时，这一轮实际使用的模型与生成参数：会话选的模型优先于
 * Agent Profile，Profile 没有给出的生成参数取模型定义的默认值（与后端的叠加顺序一致）。
 */
export function effectiveModelParams(
  sessionModelId: string | undefined,
  agent: AgentModelConfig | null | undefined,
  models: RegisteredModel[],
): EffectiveParam[] {
  const modelId = sessionModelId || agent?.model_name || models.find((m) => m.is_default)?.id;
  const modelSource: EffectiveParam['source'] = sessionModelId ? '本次会话' : agent?.model_name ? '来自 Agent' : '模型默认';
  const model = models.find((m) => m.id === modelId);

  const fromAgentOrModel = (label: string, agentValue: number | null | undefined, modelValue: number | undefined): EffectiveParam => {
    if (agentValue != null) return { label, value: String(agentValue), source: '来自 Agent' };
    if (modelValue !== undefined) return { label, value: String(modelValue), source: '模型默认' };
    return { label, value: '—', source: '未知' };
  };

  return [
    { label: '模型', value: modelId ?? '—', source: modelId ? modelSource : '未知' },
    fromAgentOrModel('温度', agent?.temperature, model?.temperature),
    fromAgentOrModel('Top P', agent?.top_p, model?.top_p),
    fromAgentOrModel('最大输出', undefined, model?.max_tokens),
  ];
}
