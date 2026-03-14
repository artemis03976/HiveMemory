import React from 'react';
import { CategorySection } from './CategorySection';
import {
  NumberInput,
  ToggleSwitch,
  SliderInput,
  FilePathInput,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface LifecycleCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const LifecycleCategories: React.FC<LifecycleCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="生命周期管理"
      paramCount={21}
      accentColor="hsl(40, 90%, 55%)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">全局</h4>
        <NumberInput
          label="高水位线"
          value={config.lifecycle.high_watermark}
          onChange={(v) => updateConfig('lifecycle.high_watermark', v)}
          min={0}
          hint="记忆保留的活力阈值"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">活力计算器</h4>
        <SliderInput
          label="代码片段权重"
          value={config.lifecycle.vitality_calculator.code_snippet_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.code_snippet_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.code_snippet_weight')}
        />
        <SliderInput
          label="事实权重"
          value={config.lifecycle.vitality_calculator.fact_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.fact_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.fact_weight')}
        />
        <SliderInput
          label="URL 资源权重"
          value={config.lifecycle.vitality_calculator.url_resource_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.url_resource_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.url_resource_weight')}
        />
        <SliderInput
          label="反思权重"
          value={config.lifecycle.vitality_calculator.reflection_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.reflection_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.reflection_weight')}
        />
        <SliderInput
          label="用户资料权重"
          value={config.lifecycle.vitality_calculator.user_profile_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.user_profile_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.user_profile_weight')}
        />
        <SliderInput
          label="进行中工作权重"
          value={config.lifecycle.vitality_calculator.work_in_progress_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.work_in_progress_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.work_in_progress_weight')}
        />
        <SliderInput
          label="默认权重"
          value={config.lifecycle.vitality_calculator.default_weight}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.default_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('lifecycle.vitality_calculator.default_weight')}
        />
        <NumberInput
          label="最大访问提升"
          value={config.lifecycle.vitality_calculator.max_access_boost}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.max_access_boost', v)}
          min={0}
        />
        <NumberInput
          label="每次访问点数"
          value={config.lifecycle.vitality_calculator.points_per_access}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.points_per_access', v)}
          min={0}
        />
        <SliderInput
          label="衰减 Lambda"
          value={config.lifecycle.vitality_calculator.decay_lambda}
          onChange={(v) => updateConfig('lifecycle.vitality_calculator.decay_lambda', v)}
          min={0}
          max={0.1}
          step={0.001}
          error={getFieldError('lifecycle.vitality_calculator.decay_lambda')}
          hint="时间衰减因子 (0-0.1)"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">强化引擎</h4>
        <ToggleSwitch
          label="启用事件历史"
          value={config.lifecycle.reinforcement.enable_event_history}
          onChange={(v) => updateConfig('lifecycle.reinforcement.enable_event_history', v)}
        />
        <NumberInput
          label="事件历史限制"
          value={config.lifecycle.reinforcement.event_history_limit}
          onChange={(v) => updateConfig('lifecycle.reinforcement.event_history_limit', v)}
          min={0}
        />
        <NumberInput
          label="命中提升"
          value={config.lifecycle.reinforcement.hit_boost}
          onChange={(v) => updateConfig('lifecycle.reinforcement.hit_boost', v)}
        />
        <NumberInput
          label="引用提升"
          value={config.lifecycle.reinforcement.citation_boost}
          onChange={(v) => updateConfig('lifecycle.reinforcement.citation_boost', v)}
        />
        <NumberInput
          label="正面反馈提升"
          value={config.lifecycle.reinforcement.positive_feedback_boost}
          onChange={(v) => updateConfig('lifecycle.reinforcement.positive_feedback_boost', v)}
        />
        <NumberInput
          label="负面反馈惩罚"
          value={config.lifecycle.reinforcement.negative_feedback_penalty}
          onChange={(v) => updateConfig('lifecycle.reinforcement.negative_feedback_penalty', v)}
        />
        <SliderInput
          label="负面置信度乘数"
          value={config.lifecycle.reinforcement.negative_confidence_multiplier}
          onChange={(v) => updateConfig('lifecycle.reinforcement.negative_confidence_multiplier', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('lifecycle.reinforcement.negative_confidence_multiplier')}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">归档器</h4>
        <FilePathInput
          label="归档目录"
          value={config.lifecycle.archiver.archive_dir}
          onChange={(v) => updateConfig('lifecycle.archiver.archive_dir', v)}
        />
        <ToggleSwitch
          label="压缩"
          value={config.lifecycle.archiver.compression}
          onChange={(v) => updateConfig('lifecycle.archiver.compression', v)}
          hint="压缩归档的记忆"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">垃圾回收器</h4>
        <NumberInput
          label="低水位线"
          value={config.lifecycle.garbage_collector.low_watermark}
          onChange={(v) => updateConfig('lifecycle.garbage_collector.low_watermark', v)}
          min={0}
          hint="垃圾回收的活力阈值"
        />
        <NumberInput
          label="批处理大小"
          value={config.lifecycle.garbage_collector.batch_size}
          onChange={(v) => updateConfig('lifecycle.garbage_collector.batch_size', v)}
          min={1}
        />
        <ToggleSwitch
          label="启用计划任务"
          value={config.lifecycle.garbage_collector.enable_schedule}
          onChange={(v) => updateConfig('lifecycle.garbage_collector.enable_schedule', v)}
        />
        <NumberInput
          label="间隔小时数"
          value={config.lifecycle.garbage_collector.interval_hours}
          onChange={(v) => updateConfig('lifecycle.garbage_collector.interval_hours', v)}
          min={1}
        />
      </div>
    </CategorySection>
  );
};
