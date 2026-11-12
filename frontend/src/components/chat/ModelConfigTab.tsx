import { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import RangeSlider from '../common/RangeSlider';
import { Toggle } from '../common/FormControls';
import { useChatRuntimeConfigStore, useChatStore } from '@/stores';
import { useAgents } from '@/hooks/useAgents';
import { effectiveModelParams } from '@/utils/effectiveModelParams';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

/**
 * 本会话的模型配置：模型可以单独选择；生成参数打开开关后才为本会话单独设置，
 * 关闭时列出沿用 Agent 与模型定义后当前生效的值。
 */
export default function ModelConfigTab() {
  const { modelSelection, updateModelSelection, overrideParams, setOverrideParams } = useChatRuntimeConfigStore();
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const currentAgentId = useChatStore((s) => s.currentAgentId);
  const { rawAgents } = useAgents();
  const agentConfig = rawAgents.find((a) => a.alias === currentAgentId)?.agent_config;
  const effective = effectiveModelParams(modelSelection.model_id, agentConfig, models);
  const agentModel = effectiveModelParams(undefined, agentConfig, models).find((p) => p.label === '模型')?.value;

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  return (
    <div className="flex flex-col gap-3 rounded-[14px] border border-line bg-surface-container p-3">
      <label className="flex items-center gap-2">
        <span className="w-16 shrink-0 text-[13px] text-fg-2">模型</span>
        <span className="relative min-w-0 flex-1">
          <select
            value={modelSelection.model_id}
            onChange={(e) => updateModelSelection({ model_id: e.target.value })}
            className="h-8 w-full cursor-pointer appearance-none rounded-lg border border-line bg-field pl-2.5 pr-8 text-[13px] text-fg focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            {/* 空值表示跟随 Agent Profile 的模型 */}
            <option value="">{agentModel && agentModel !== '—' ? `跟随 Agent（${agentModel}）` : '跟随 Agent'}</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name || m.id}{m.is_default ? '（默认）' : ''}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-3" aria-hidden="true" />
        </span>
      </label>

      <div className="flex items-center gap-2.5 border-t border-line pt-3">
        <div className="flex flex-1 flex-col gap-0.5">
          <span className="text-sm font-semibold text-fg">单独设置生成参数</span>
          <span className="text-xs text-fg-3">对之后发出的每一轮生效；关闭时沿用 Agent 与模型定义</span>
        </div>
        <Toggle checked={overrideParams} onChange={setOverrideParams} label="单独设置生成参数" />
      </div>

      {overrideParams ? (
        <div className="flex flex-col gap-5 pt-1">
          <RangeSlider
            label="温度"
            min={0}
            max={2}
            step={0.1}
            defaultValue={1.0}
            value={modelSelection.temperature}
            onChange={(value) => updateModelSelection({ temperature: value })}
          />
          <RangeSlider
            label="Top P"
            min={0}
            max={1}
            step={0.05}
            defaultValue={1}
            value={modelSelection.top_p}
            onChange={(value) => updateModelSelection({ top_p: value })}
          />
          <RangeSlider
            label="最大输出"
            min={256}
            max={32768}
            step={256}
            defaultValue={32768}
            value={modelSelection.max_tokens}
            onChange={(value) => updateModelSelection({ max_tokens: value })}
          />
        </div>
      ) : (
        <>
          <span className="text-xs font-semibold text-fg-3">当前生效</span>
          {effective.map((param) => (
            <div key={param.label} className="flex items-baseline gap-2 text-[13px]">
              <span className="w-16 shrink-0 text-fg-2">{param.label}</span>
              <span className="min-w-0 flex-1 truncate font-mono text-fg">{param.value}</span>
              <span className="shrink-0 text-xs text-fg-3">{param.source}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
