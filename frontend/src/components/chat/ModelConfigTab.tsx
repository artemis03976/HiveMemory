import { Cpu, SlidersHorizontal, ChevronRight } from 'lucide-react';
import RangeSlider from '../common/RangeSlider';
import { useChatRuntimeConfigStore } from '@/stores/chatRuntimeConfigStore';

export default function ModelConfigTab() {
  const { generationOptions, updateGenerationOptions } = useChatRuntimeConfigStore();

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
            <option value="deepseek/deepseek-chat" className="bg-surface-container">DeepSeek Chat</option>
            <option value="gpt-4o" className="bg-surface-container">GPT-4o</option>
            <option value="claude-3-5" className="bg-surface-container">Claude 3.5 Sonnet</option>
            <option value="gemini-pro" className="bg-surface-container">Gemini 1.5 Pro</option>
          </select>
          <ChevronRight className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none rotate-90" />
        </div>
      </div>

      {/* 参数调节 */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-slate-300 font-bold text-[13px]">
          <SlidersHorizontal className="w-4 h-4 text-primary" />生成参数
        </div>
        
        <div className="space-y-5 p-4 rounded-xl bg-black/20 border border-white/5 ghost-border">
          <RangeSlider
            label="温度 (Temperature)"
            min={0}
            max={2}
            step={0.1}
            defaultValue={0.7}
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
            max={8192}
            step={256}
            defaultValue={4096}
            value={generationOptions.max_tokens}
            onChange={(value) => updateGenerationOptions({ max_tokens: value })}
          />
        </div>
      </div>

      {/* <div className="pt-4 flex justify-center">
        <button className="flex items-center gap-2 px-6 py-2 bg-primary/10 text-primary rounded-full text-xs font-bold tracking-wider hover:bg-primary/20 hover:shadow-[0_0_15px_rgba(197,154,255,0.3)] transition-all ghost-border">
          <Save className="w-3.5 h-3.5" />
          保存当前配置
        </button>
      </div> */}
    </div>
  );
}
