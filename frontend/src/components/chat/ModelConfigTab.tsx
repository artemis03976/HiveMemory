import { useEffect, useState } from 'react';
import { Cpu, SlidersHorizontal, ChevronRight } from 'lucide-react';
import RangeSlider from '../common/RangeSlider';
import { Toggle } from '../common/FormControls';
import { useChatRuntimeConfigStore } from '@/stores';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

export default function ModelConfigTab() {
  const { generationOptions, updateGenerationOptions, overrideParams, setOverrideParams } =
    useChatRuntimeConfigStore();
  const [models, setModels] = useState<RegisteredModel[]>([]);

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
        <div className="flex items-center gap-2 text-slate-300 font-bold text-[13px]">
          <Cpu className="w-4 h-4 text-primary" />模型引擎
        </div>

        <div className="relative">
          <select
            value={generationOptions.model}
            onChange={(e) => updateGenerationOptions({ model: e.target.value })}
            className="w-full bg-black/20 border border-white/10 rounded-xl p-3 text-sm text-slate-300 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all appearance-none cursor-pointer ghost-border"
          >
            {/* 空值 = 跟随 Agent Profile 的默认模型 */}
            <option value="" className="bg-surface-container">跟随 Agent 默认</option>
            {models.map((m) => (
              <option key={m.id} value={m.id} className="bg-surface-container">
                {m.display_name}{m.is_default ? '（默认）' : ''}
              </option>
            ))}
          </select>
          <ChevronRight className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none rotate-90" />
        </div>
      </div>

      {/* 参数调节 */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-300 font-bold text-[13px]">
            <SlidersHorizontal className="w-4 h-4 text-primary" />生成参数
          </div>
          {/* 总开关：关闭时跟随 profile/模型定义默认，不下发这些参数 */}
          <Toggle checked={overrideParams} onChange={setOverrideParams} />
        </div>

        {!overrideParams && (
          <p className="text-[11px] text-slate-500 leading-relaxed">
            当前跟随 Agent / 模型默认参数。打开开关可为本次会话自定义覆盖。
          </p>
        )}

        <div
          className={`space-y-5 p-4 rounded-xl bg-black/20 border border-white/5 ghost-border transition-opacity ${
            overrideParams ? '' : 'opacity-40 pointer-events-none select-none'
          }`}
        >
          <RangeSlider
            label="温度 (Temperature)"
            min={0}
            max={2}
            step={0.1}
            defaultValue={1.0}
            value={generationOptions.temperature}
            onChange={(value) => updateGenerationOptions({ temperature: value })}
          />
          <RangeSlider
            label="Top P 采样率"
            min={0}
            max={1}
            step={0.05}
            defaultValue={1}
            value={generationOptions.top_p}
            onChange={(value) => updateGenerationOptions({ top_p: value })}
          />
          <RangeSlider
            label="最大生成长度 (Max Tokens)"
            min={256}
            max={32768}
            step={256}
            defaultValue={32768}
            value={generationOptions.max_tokens}
            onChange={(value) => updateGenerationOptions({ max_tokens: value })}
          />
        </div>
      </div>
    </div>
  );
}
