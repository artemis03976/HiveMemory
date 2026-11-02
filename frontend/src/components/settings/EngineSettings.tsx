import { useEffect, useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { SettingSection, SettingRow, Input, NullableNumberInput } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';
import type { ModelSelection } from '@/types/config';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';

/**
 * 内部引擎配置 — Gateway（gateway.llm）、帕秋莉引擎（patchouli.llm）与默认 embedding 模型。
 *
 * 两个组件各有一份模型选择：model_id 引用注册表中的模型（模型标识与凭证统一管理），
 * temperature/top_p/max_tokens 覆盖模型定义的默认值，留空表示沿用。组件在启动时
 * 解析模型，保存后重启生效。llm 为 null 表示组件不使用 LLM，只能在 config.yaml 中设置：
 * 页面不知道各组件的默认生成参数，从这里重新开启会丢掉它们。
 */
export function EngineSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  const [models, setModels] = useState<RegisteredModel[]>([]);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  const modelSelector = (path: string, value: string | null) =>
    models.length > 0 ? (
      <div className="relative">
        <select
          value={value ?? ''}
          onChange={(e) => updateConfig(path, e.target.value || null)}
          className="bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 transition-all cursor-pointer font-mono outline-none w-64 appearance-none"
        >
          <option value="" className="bg-surface-container">注册表默认模型</option>
          {models.map((m) => (
            <option key={m.id} value={m.id} className="bg-surface-container">
              {`${m.display_name}${m.is_default ? '（默认）' : ''}`}
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
        placeholder="注册表 ID，留空为默认模型"
      />
    );

  const selectionRows = (
    prefix: string,
    selection: ModelSelection | null,
    descriptions: { model: string; temperature?: string; max_tokens?: string },
  ) => {
    if (selection === null) {
      return (
        <p className="text-sm text-slate-400 py-2">
          已配置为不使用 LLM。需要启用时，在 <code className="font-mono">configs/config.yaml</code> 中设置{' '}
          <code className="font-mono">{prefix}</code> 并重启服务。
        </p>
      );
    }
    return (
      <>
        <SettingRow label="模型" description={descriptions.model}>
          {modelSelector(`${prefix}.model_id`, selection.model_id)}
        </SettingRow>
        <SettingRow label="温度" description={descriptions.temperature ?? '采样温度 (0.0 到 2.0)，留空沿用模型默认值。'}>
          <NullableNumberInput
            value={selection.temperature}
            onChange={(v) => updateConfig(`${prefix}.temperature`, v)}
            step="0.1"
            className="w-24"
            placeholder="模型默认"
            error={getFieldError(`${prefix}.temperature`)}
          />
        </SettingRow>
        <SettingRow label="Top-p" description="核采样阈值 (0.0 到 1.0)，留空沿用模型默认值。">
          <NullableNumberInput
            value={selection.top_p}
            onChange={(v) => updateConfig(`${prefix}.top_p`, v)}
            step="0.05"
            className="w-24"
            placeholder="模型默认"
          />
        </SettingRow>
        <SettingRow label="最大 Token 数" description={descriptions.max_tokens ?? '生成的最大 Token 数量，留空沿用模型默认值。'}>
          <NullableNumberInput
            value={selection.max_tokens}
            onChange={(v) => updateConfig(`${prefix}.max_tokens`, v)}
            className="w-32"
            placeholder="模型默认"
          />
        </SettingRow>
      </>
    );
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="LLM: 全局网关">
        {selectionRows('gateway.llm', config.gateway.llm, {
          model: '从注册表中选择网关使用的模型。保存后重启生效。',
          temperature: '采样温度 (0.0 到 2.0)。网关建议使用极低值以稳定路由；留空沿用模型默认值。',
        })}
      </SettingSection>

      <SettingSection title="LLM: 帕秋莉的底层引擎">
        {selectionRows('patchouli.llm', config.patchouli.llm, {
          model: '从注册表中选择帕秋莉引擎使用的模型。保存后重启生效。',
          max_tokens: '一次结算输出多条记忆，输出被截断时任务失败，需要较大的输出上限；留空沿用模型默认值。',
        })}
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
