import { SettingSection, SettingRow, Input, Select } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function ModelSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="LLM: 全局网关">
        <SettingRow label="提供商" description="网关模型的 API 提供商。">
          <Input value={config.llm.gateway.provider} onChange={(v: string) => updateConfig('llm.gateway.provider', v)} />
        </SettingRow>
        <SettingRow label="模型" description="模型标识符。">
          <Input value={config.llm.gateway.model} onChange={(v: string) => updateConfig('llm.gateway.model', v)} className="w-64" />
        </SettingRow>
        <SettingRow label="API 基础 URL" description="自定义 API 端点（可选）。">
          <Input value={config.llm.gateway.api_base || ''} onChange={(v: string) => updateConfig('llm.gateway.api_base', v || null)} className="w-64" placeholder="可选" />
        </SettingRow>
        <SettingRow label="温度" description="采样温度 (0.0 到 2.0)。">
          <Input type="number" value={config.llm.gateway.temperature} onChange={(v: number) => updateConfig('llm.gateway.temperature', v)} step="0.1" className="w-24" error={getFieldError('llm.gateway.temperature')} />
        </SettingRow>
        <SettingRow label="最大 Token 数" description="生成的最大 Token 数量。">
          <Input type="number" value={config.llm.gateway.max_tokens} onChange={(v: number) => updateConfig('llm.gateway.max_tokens', v)} className="w-32" />
        </SettingRow>
      </SettingSection>

      <SettingSection title="LLM: 帕秋莉的底层引擎">
        <SettingRow label="提供商">
          <Input value={config.llm.librarian.provider} onChange={(v: string) => updateConfig('llm.librarian.provider', v)} />
        </SettingRow>
        <SettingRow label="模型">
          <Input value={config.llm.librarian.model} onChange={(v: string) => updateConfig('llm.librarian.model', v)} className="w-64" />
        </SettingRow>
        <SettingRow label="温度">
          <Input type="number" value={config.llm.librarian.temperature} onChange={(v: number) => updateConfig('llm.librarian.temperature', v)} step="0.1" className="w-24" error={getFieldError('llm.librarian.temperature')} />
        </SettingRow>
      </SettingSection>

      <SettingSection title="Embedding: 默认 embedding 模型">
        <SettingRow label="模型名称" description="用于生成向量嵌入的模型。">
          <Input value={config.embedding.default.model_name} onChange={(v: string) => updateConfig('embedding.default.model_name', v)} className="w-64" />
        </SettingRow>
        <SettingRow label="计算设备" description="计算设备 (cpu, cuda, mps)。">
          <Select 
            value={config.embedding.default.device}
            onChange={(v: string) => updateConfig('embedding.default.device', v)}
            error={getFieldError('embedding.default.device')}
            options={[
              {label: 'CPU', value: 'cpu'},
              {label: 'CUDA', value: 'cuda'},
              {label: 'MPS', value: 'mps'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="维度" description="向量维度大小。">
          <Input type="number" value={config.embedding.default.dimension} onChange={(v: number) => updateConfig('embedding.default.dimension', v)} className="w-32" />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
