import React from 'react';
import { CategorySection } from './CategorySection';
import {
  NumberInput,
  ToggleSwitch,
  SelectDropdown,
  TagInput,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface GatewayCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const GatewayCategories: React.FC<GatewayCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="网关设置"
      paramCount={9}
      accentColor="hsl(45, 100%, 60%)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">拦截器</h4>
        <ToggleSwitch
          label="启用"
          value={config.gateway.interceptor.enabled}
          onChange={(v) => updateConfig('gateway.interceptor.enabled', v)}
        />
        <ToggleSwitch
          label="启用系统拦截"
          value={config.gateway.interceptor.enable_system}
          onChange={(v) => updateConfig('gateway.interceptor.enable_system', v)}
        />
        <ToggleSwitch
          label="启用聊天拦截"
          value={config.gateway.interceptor.enable_chat}
          onChange={(v) => updateConfig('gateway.interceptor.enable_chat', v)}
        />
        <TagInput
          label="自定义系统模式"
          value={config.gateway.interceptor.custom_system_patterns}
          onChange={(v) => updateConfig('gateway.interceptor.custom_system_patterns', v)}
          placeholder="添加正则模式"
          hint="用于系统消息拦截的正则表达式模式"
        />
        <TagInput
          label="自定义聊天模式"
          value={config.gateway.interceptor.custom_chat_patterns}
          onChange={(v) => updateConfig('gateway.interceptor.custom_chat_patterns', v)}
          placeholder="添加正则模式"
          hint="用于聊天消息拦截的正则表达式模式"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">分析器</h4>
        <ToggleSwitch
          label="启用"
          value={config.gateway.analyzer.enabled}
          onChange={(v) => updateConfig('gateway.analyzer.enabled', v)}
        />
        <NumberInput
          label="上下文窗口"
          value={config.gateway.analyzer.context_window}
          onChange={(v) => updateConfig('gateway.analyzer.context_window', v)}
          min={1}
          max={10}
          error={getFieldError('gateway.analyzer.context_window')}
          hint="要分析的最近消息数量 (1-10)"
        />
        <SelectDropdown
          label="提示词变体"
          value={config.gateway.analyzer.prompt_variant}
          onChange={(v) => updateConfig('gateway.analyzer.prompt_variant', v)}
          options={[{ value: 'default', label: '默认' }]}
        />
        <SelectDropdown
          label="提示词语言"
          value={config.gateway.analyzer.prompt_language}
          onChange={(v) => updateConfig('gateway.analyzer.prompt_language', v)}
          options={[
            { value: 'zh', label: '中文' },
            { value: 'en', label: '英文' },
          ]}
        />
      </div>
    </CategorySection>
  );
};
