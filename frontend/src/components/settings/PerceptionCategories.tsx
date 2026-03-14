import React from 'react';
import { CategorySection } from './CategorySection';
import {
  NumberInput,
  ToggleSwitch,
  SelectDropdown,
  SliderInput,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface PerceptionCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const PerceptionCategories: React.FC<PerceptionCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="感知层"
      paramCount={13}
      accentColor="hsl(150, 80%, 45%)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">基础配置</h4>
        <NumberInput
          label="空闲超时 (秒)"
          value={config.perception.engine.idle_timeout_seconds}
          onChange={(v) => updateConfig('perception.engine.idle_timeout_seconds', v)}
          min={0}
        />
        <NumberInput
          label="扫描间隔 (秒)"
          value={config.perception.engine.scan_interval_seconds}
          onChange={(v) => updateConfig('perception.engine.scan_interval_seconds', v)}
          min={1}
        />
        <NumberInput
          label="折叠 Token 阈值"
          value={config.perception.engine.fold_token_threshold}
          onChange={(v) => updateConfig('perception.engine.fold_token_threshold', v)}
          min={0}
        />
        <NumberInput
          label="折叠保留最近块数"
          value={config.perception.engine.fold_retain_recent_blocks}
          onChange={(v) => updateConfig('perception.engine.fold_retain_recent_blocks', v)}
          min={0}
        />
        <NumberInput
          label="最大驻留主题数"
          value={config.perception.engine.max_resident_topics}
          onChange={(v) => updateConfig('perception.engine.max_resident_topics', v)}
          min={1}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">中继控制器</h4>
        <SelectDropdown
          label="引擎类型"
          value={config.perception.engine.relay.engine.type}
          onChange={(v) => updateConfig('perception.engine.relay.engine.type', v)}
          options={[
            { value: 'simple', label: '简单' },
            { value: 'llm', label: '基于 LLM' },
          ]}
          error={getFieldError('perception.engine.relay.engine.type')}
        />
        <NumberInput
          label="最大处理 Token 数"
          value={config.perception.engine.relay.engine.max_processing_tokens}
          onChange={(v) => updateConfig('perception.engine.relay.engine.max_processing_tokens', v)}
          min={0}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">语义吸附器</h4>
        <SliderInput
          label="高语义阈值"
          value={config.perception.engine.adsorber.semantic_threshold_high}
          onChange={(v) => updateConfig('perception.engine.adsorber.semantic_threshold_high', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('perception.engine.adsorber.semantic_threshold_high')}
        />
        <SliderInput
          label="低语义阈值"
          value={config.perception.engine.adsorber.semantic_threshold_low}
          onChange={(v) => updateConfig('perception.engine.adsorber.semantic_threshold_low', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('perception.engine.adsorber.semantic_threshold_low')}
        />
        <NumberInput
          label="短文本阈值"
          value={config.perception.engine.adsorber.short_text_threshold}
          onChange={(v) => updateConfig('perception.engine.adsorber.short_text_threshold', v)}
          min={0}
        />
        <SliderInput
          label="EMA Alpha"
          value={config.perception.engine.adsorber.ema_alpha}
          onChange={(v) => updateConfig('perception.engine.adsorber.ema_alpha', v)}
          min={0.01}
          max={1}
          step={0.01}
          error={getFieldError('perception.engine.adsorber.ema_alpha')}
          hint="必须在 (0, 1] 范围内"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">仲裁器</h4>
        <ToggleSwitch
          label="启用"
          value={config.perception.engine.adsorber.arbiter.enabled}
          onChange={(v) => updateConfig('perception.engine.adsorber.arbiter.enabled', v)}
        />
        <SelectDropdown
          label="引擎类型"
          value={config.perception.engine.adsorber.arbiter.engine.type}
          onChange={(v) => updateConfig('perception.engine.adsorber.arbiter.engine.type', v)}
          options={[
            { value: 'reranker', label: '重排序器' },
            { value: 'slm', label: '小型语言模型' },
          ]}
          error={getFieldError('perception.engine.adsorber.arbiter.engine.type')}
        />
        <SliderInput
          label="阈值"
          value={config.perception.engine.adsorber.arbiter.engine.threshold}
          onChange={(v) => updateConfig('perception.engine.adsorber.arbiter.engine.threshold', v)}
          min={-10}
          max={0}
          step={0.1}
          error={getFieldError('perception.engine.adsorber.arbiter.engine.threshold')}
          hint="范围: -10 到 0"
        />
      </div>
    </CategorySection>
  );
};
