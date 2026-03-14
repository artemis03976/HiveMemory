import React from 'react';
import { CategorySection } from './CategorySection';
import {
  ToggleSwitch,
  SliderInput,
  TextArea,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface GenerationCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const GenerationCategories: React.FC<GenerationCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="记忆生成"
      paramCount={7}
      accentColor="hsl(340, 80%, 60%)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">提取器</h4>
        <ToggleSwitch
          label="启用"
          value={config.generation.extractor.enabled}
          onChange={(v) => updateConfig('generation.extractor.enabled', v)}
        />
        <TextArea
          label="系统提示词"
          value={config.generation.extractor.system_prompt}
          onChange={(v) => updateConfig('generation.extractor.system_prompt', v)}
          placeholder="留空以使用默认提示词"
          rows={4}
        />
        <TextArea
          label="用户提示词"
          value={config.generation.extractor.user_prompt}
          onChange={(v) => updateConfig('generation.extractor.user_prompt', v)}
          placeholder="留空以使用默认提示词"
          rows={4}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">去重器</h4>
        <ToggleSwitch
          label="启用"
          value={config.generation.deduplicator.enabled}
          onChange={(v) => updateConfig('generation.deduplicator.enabled', v)}
        />
        <SliderInput
          label="高相似度阈值"
          value={config.generation.deduplicator.high_similarity_threshold}
          onChange={(v) => updateConfig('generation.deduplicator.high_similarity_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('generation.deduplicator.high_similarity_threshold')}
        />
        <SliderInput
          label="低相似度阈值"
          value={config.generation.deduplicator.low_similarity_threshold}
          onChange={(v) => updateConfig('generation.deduplicator.low_similarity_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('generation.deduplicator.low_similarity_threshold')}
        />
        <SliderInput
          label="内容相似度阈值"
          value={config.generation.deduplicator.content_similarity_threshold}
          onChange={(v) => updateConfig('generation.deduplicator.content_similarity_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('generation.deduplicator.content_similarity_threshold')}
        />
        <ToggleSwitch
          label="启用活力追踪"
          value={config.generation.deduplicator.enable_vitality_tracking}
          onChange={(v) => updateConfig('generation.deduplicator.enable_vitality_tracking', v)}
        />
      </div>
    </CategorySection>
  );
};
