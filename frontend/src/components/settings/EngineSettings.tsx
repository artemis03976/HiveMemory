import { useEffect, useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { SettingSection, SettingRow, Input } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

/**
 * 内部引擎配置 — Gateway（gateway.llm）、帕秋莉引擎（patchouli.llm）与默认 embedding 模型。
 *
 * 两个组件以 model_id 引用注册表中的模型（模型标识与凭证统一管理），
 * temperature/max_tokens 是组件任务要求的生成参数，覆盖注册表模型的默认值。
 * 组件在启动时解析模型，保存后重启生效。
 */
export function EngineSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  const [models, setModels] = useState<RegisteredModel[]>([]);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  // 下拉选项：注册表默认模型与注册表模型列表；空值表示组件不使用 LLM
  const modelOptions = [
    { label: '注册表默认模型', value: 'default' },
    ...models.map((m) => ({
      label: `${m.display_name}${m.is_default ? '（默认）' : ''}`,
      value: m.id,
    })),
  ];

  const modelSelector = (path: string, value: string | null) =>
    models.length > 0 ? (
      <div className="relative">
        <select
          value={value ?? ''}
          onChange={(e) => updateConfig(path, e.target.value || null)}
          className="bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all cursor-pointer font-mono outline-none w-64 appearance-none"
        >
          <option value="" className="bg-surface-container">— 不使用 LLM —</option>
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
        value={value ?? ''}
        onChange={(v: string) => updateConfig(path, v || null)}
        className="w-64"
        placeholder="注册表 ID，如 deepseek-chat"
      />
    );

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="LLM: 全局网关">
        <SettingRow
          label="模型"
          description="从注册表中选择网关使用的模型。保存后重启生效。"
        >
          {modelSelector('gateway.llm.model_id', config.gateway.llm.model_id)}
        </SettingRow>
        <SettingRow label="温度" description="采样温度 (0.0 到 2.0)。网关建议使用极低值以稳定路由。">
          <Input
            type="number"
            value={config.gateway.llm.temperature}
            onChange={(v: number) => updateConfig('gateway.llm.temperature', v)}
            step="0.1"
            className="w-24"
            error={getFieldError('gateway.llm.temperature')}
          />
        </SettingRow>
        <SettingRow label="最大 Token 数" description="生成的最大 Token 数量。">
          <Input
            type="number"
            value={config.gateway.llm.max_tokens}
            onChange={(v: number) => updateConfig('gateway.llm.max_tokens', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>

      <SettingSection title="LLM: 帕秋莉的底层引擎">
        <SettingRow label="模型" description="从注册表中选择帕秋莉引擎使用的模型。保存后重启生效。">
          {modelSelector('patchouli.llm.model_id', config.patchouli.llm.model_id)}
        </SettingRow>
        <SettingRow label="温度">
          <Input
            type="number"
            value={config.patchouli.llm.temperature}
            onChange={(v: number) => updateConfig('patchouli.llm.temperature', v)}
            step="0.1"
            className="w-24"
            error={getFieldError('patchouli.llm.temperature')}
          />
        </SettingRow>
        <SettingRow
          label="最大 Token 数"
          description="一次结算输出多条记忆，输出被截断时任务失败，需要较大的输出上限。"
        >
          <Input
            type="number"
            value={config.patchouli.llm.max_tokens}
            onChange={(v: number) => updateConfig('patchouli.llm.max_tokens', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>

      <SettingSection title="Embedding: 默认 embedding 模型">
        <SettingRow label="模型名称" description="用于生成向量嵌入的模型。">
          <Input
            value={config.shared.embedding.default.model_name}
            onChange={(v: string) => updateConfig('shared.embedding.default.model_name', v)}
            className="w-64"
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
