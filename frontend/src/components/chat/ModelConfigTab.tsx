import { useEffect, useState } from 'react';
import { Cpu, SlidersHorizontal, ChevronRight } from 'lucide-react';
import RangeSlider from '../common/RangeSlider';
import { Toggle } from '../common/FormControls';
import { useChatRuntimeConfigStore, useChatStore } from '@/stores';
import { useAgents } from '@/hooks/useAgents';
import { effectiveModelParams } from '@/utils/effectiveModelParams';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

export default function ModelConfigTab() {
  const { modelSelection, updateModelSelection, overrideParams, setOverrideParams } =
    useChatRuntimeConfigStore();
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const currentAgentId = useChatStore((s) => s.currentAgentId);
  const { rawAgents } = useAgents();
  const agentConfig = rawAgents.find((a) => a.alias === currentAgentId)?.agent_config;
  const effective = effectiveModelParams(modelSelection.model_id, agentConfig, models);

  // 加载注册表模型列表，供会话级模型覆盖选择
  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  return (
    <div className="px-2 space-y-6 mt-2">
      {/* 基础模型设置 */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-fg-2 font-bold text-[13px]">
          <Cpu className="w-4 h-4 text-primary" />模型引擎
        </div>

        <div className="relative">
          <select
            value={modelSelection.model_id}
            onChange={(e) => updateModelSelection({ model_id: e.target.value })}
            className="w-full bg-field border border-line rounded-xl p-3 text-sm text-fg-2 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all appearance-none cursor-pointer border border-line"
          >
            {/* 空值 = 跟随 Agent Profile 的默认模型 */}
            <option value="" className="bg-surface-container">跟随 Agent 默认</option>
            {models.map((m) => (
              <option key={m.id} value={m.id} className="bg-surface-container">
                {m.display_name}{m.is_default ? '（默认）' : ''}
              </option>
            ))}
          </select>
          <ChevronRight className="w-4 h-4 text-fg-3 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none rotate-90" />
        </div>
      </div>

      {/* 参数调节 */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-fg-2 font-bold text-[13px]">
            <SlidersHorizontal className="w-4 h-4 text-primary" />生成参数
          </div>
          {/* 总开关：关闭时跟随 profile/模型定义默认，不下发这些参数 */}
          <Toggle checked={overrideParams} onChange={setOverrideParams} />
        </div>

        {!overrideParams && (
          <div className="space-y-2.5 rounded-xl border border-line bg-field p-4">
            <p className="text-xs text-fg-3">当前生效（打开开关可为本次会话单独设置）</p>
            {effective.map((param) => (
              <div key={param.label} className="flex items-baseline gap-2 text-[13px]">
                <span className="w-16 shrink-0 text-fg-2">{param.label}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-fg">{param.value}</span>
                <span className="shrink-0 text-xs text-fg-3">{param.source}</span>
              </div>
            ))}
          </div>
        )}

        {overrideParams && (
        <div className="space-y-5 rounded-xl border border-line bg-field p-4">
          <RangeSlider
            label="温度 (Temperature)"
            min={0}
            max={2}
            step={0.1}
            defaultValue={1.0}
            value={modelSelection.temperature}
            onChange={(value) => updateModelSelection({ temperature: value })}
          />
          <RangeSlider
            label="Top P 采样率"
            min={0}
            max={1}
            step={0.05}
            defaultValue={1}
            value={modelSelection.top_p}
            onChange={(value) => updateModelSelection({ top_p: value })}
          />
          <RangeSlider
            label="最大生成长度 (Max Tokens)"
            min={256}
            max={32768}
            step={256}
            defaultValue={32768}
            value={modelSelection.max_tokens}
            onChange={(value) => updateModelSelection({ max_tokens: value })}
          />
        </div>
        )}
      </div>
    </div>
  );
}
