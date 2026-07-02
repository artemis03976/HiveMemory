import { useEffect, useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { SettingSection, SettingRow, Input, Select } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

/**
 * 内部引擎配置 — gateway（全局网关）/ librarian（帕秋莉引擎）/ embedding。
 *
 * gateway/librarian 使用 model_id 引用注册表中的模型（模型标识与凭证统一管理），
 * temperature/max_tokens 是组件专属调参，覆盖注册表模型的默认值。
 * worker 段已由 agent profile + 注册表在运行时决定，不在此列。
 */
export function EngineSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  const [models, setModels] = useState<RegisteredModel[]>([]);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  // 下拉选项：注册表模型列表
  const modelOptions = models.map((m) => ({
    label: `${m.display_name}${m.is_default ? '（默认）' : ''}`,
    value: m.id,
  }));

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="LLM: 全局网关">
        <SettingRow
          label="模型"
          description="从注册表中选择网关使用的模型。切换后保存生效。"
        >
          {models.length > 0 ? (
            <div className="relative">
              <select
                value={config.llm.gateway.model_id ?? ''}
                onChange={(e) => updateConfig('llm.gateway.model_id', e.target.value || null)}
                className="bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all cursor-pointer font-mono outline-none w-64 appearance-none"
              >
                <option value="" className="bg-surface-container">— 选择模型 —</option>
                {modelOptions.map((o) => (
                  <option key={o.value} value={o.value} className="bg-surface-container">
                    {o.label}
                  </option>
                ))}
              </select>
              <ChevronRight className="w-3.5 h-3.5 text-slate-400 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none rotate-90" />
            </div>
          ) : (
            <Input
              value={config.llm.gateway.model_id ?? ''}
              onChange={(v: string) => updateConfig('llm.gateway.model_id', v || null)}
              className="w-64"
              placeholder="注册表 ID，如 deepseek-chat"
            />
          )}
        </SettingRow>
        <SettingRow label="温度" description="采样温度 (0.0 到 2.0)。网关建议使用极低值以稳定路由。">
          <Input
            type="number"
            value={config.llm.gateway.temperature}
            onChange={(v: number) => updateConfig('llm.gateway.temperature', v)}
            step="0.1"
            className="w-24"
            error={getFieldError('llm.gateway.temperature')}
          />
        </SettingRow>
        <SettingRow label="最大 Token 数" description="生成的最大 Token 数量。">
          <Input
            type="number"
            value={config.llm.gateway.max_tokens}
            onChange={(v: number) => updateConfig('llm.gateway.max_tokens', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>

      <SettingSection title="LLM: 帕秋莉的底层引擎">
        <SettingRow label="模型" description="从注册表中选择帕秋莉引擎使用的模型。">
          {models.length > 0 ? (
            <div className="relative">
              <select
                value={config.llm.librarian.model_id ?? ''}
                onChange={(e) => updateConfig('llm.librarian.model_id', e.target.value || null)}
                className="bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all cursor-pointer font-mono outline-none w-64 appearance-none"
              >
                <option value="" className="bg-surface-container">— 选择模型 —</option>
                {modelOptions.map((o) => (
                  <option key={o.value} value={o.value} className="bg-surface-container">
                    {o.label}
                  </option>
                ))}
              </select>
              <ChevronRight className="w-3.5 h-3.5 text-slate-400 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none rotate-90" />
            </div>
          ) : (
            <Input
              value={config.llm.librarian.model_id ?? ''}
              onChange={(v: string) => updateConfig('llm.librarian.model_id', v || null)}
              className="w-64"
              placeholder="注册表 ID，如 deepseek-chat"
            />
          )}
        </SettingRow>
        <SettingRow label="温度">
          <Input
            type="number"
            value={config.llm.librarian.temperature}
            onChange={(v: number) => updateConfig('llm.librarian.temperature', v)}
            step="0.1"
            className="w-24"
            error={getFieldError('llm.librarian.temperature')}
          />
        </SettingRow>
        <SettingRow label="最大 Token 数">
          <Input
            type="number"
            value={config.llm.librarian.max_tokens}
            onChange={(v: number) => updateConfig('llm.librarian.max_tokens', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>

      <SettingSection title="Embedding: 默认 embedding 模型">
        <SettingRow label="模型名称" description="用于生成向量嵌入的模型。">
          <Input
            value={config.embedding.default.model_name}
            onChange={(v: string) => updateConfig('embedding.default.model_name', v)}
            className="w-64"
          />
        </SettingRow>
        <SettingRow label="计算设备" description="计算设备 (cpu, cuda, mps)。">
          <Select
            value={config.embedding.default.device}
            onChange={(v: string) => updateConfig('embedding.default.device', v)}
            error={getFieldError('embedding.default.device')}
            options={[
              { label: 'CPU', value: 'cpu' },
              { label: 'CUDA', value: 'cuda' },
              { label: 'MPS', value: 'mps' },
            ]}
          />
        </SettingRow>
        <SettingRow label="维度" description="向量维度大小。">
          <Input
            type="number"
            value={config.embedding.default.dimension}
            onChange={(v: number) => updateConfig('embedding.default.dimension', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>

      <p className="text-xs text-slate-500 mt-2 leading-relaxed">
        模型选择从注册表动态加载。在
        <span className="text-primary font-medium">「模型注册表」</span>
        页管理可用模型后，此处选项自动更新。凭证由 provider 表统一管理，无需在此单独设置。
      </p>
    </div>
  );
}