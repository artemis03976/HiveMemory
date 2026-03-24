import { SettingSection, SettingRow, Input, Toggle } from '../common/FormControls';

export function LifecycleSettings({ config, updateConfig, getFieldError }: any) {
  if (!config) return null;
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="活力值计算 (Vitality Calculator)">
        <SettingRow label="代码片段权重" description="代码片段记忆的基础活力乘数。">
          <Input type="number" value={config.lifecycle.vitality_calculator.code_snippet_weight} onChange={(v: any) => updateConfig('lifecycle.vitality_calculator.code_snippet_weight', v)} step="0.1" className="w-24" error={getFieldError('lifecycle.vitality_calculator.code_snippet_weight')} />
        </SettingRow>
        <SettingRow label="事实权重" description="事实知识记忆的基础活力乘数。">
          <Input type="number" value={config.lifecycle.vitality_calculator.fact_weight} onChange={(v: any) => updateConfig('lifecycle.vitality_calculator.fact_weight', v)} step="0.1" className="w-24" error={getFieldError('lifecycle.vitality_calculator.fact_weight')} />
        </SettingRow>
        <SettingRow label="衰减 Lambda" description="活力值随时间的衰减率。">
          <Input type="number" value={config.lifecycle.vitality_calculator.decay_lambda} onChange={(v: any) => updateConfig('lifecycle.vitality_calculator.decay_lambda', v)} step="0.001" className="w-24" error={getFieldError('lifecycle.vitality_calculator.decay_lambda')} />
        </SettingRow>
      </SettingSection>

      <SettingSection title="正向强化 (Reinforcement)">
        <SettingRow label="命中加成" description="每次记忆被成功检索时增加的活力点数。">
          <Input type="number" value={config.lifecycle.reinforcement.hit_boost} onChange={(v: any) => updateConfig('lifecycle.reinforcement.hit_boost', v)} step="0.5" className="w-24" />
        </SettingRow>
        <SettingRow label="引用加成" description="当记忆被明确引用时增加的活力点数。">
          <Input type="number" value={config.lifecycle.reinforcement.citation_boost} onChange={(v: any) => updateConfig('lifecycle.reinforcement.citation_boost', v)} step="1.0" className="w-24" />
        </SettingRow>
      </SettingSection>

      <SettingSection title="垃圾回收 (Garbage Collection)">
        <SettingRow label="低水位线" description="低于此活力阈值的记忆将被归档。">
          <Input type="number" value={config.lifecycle.garbage_collector.low_watermark} onChange={(v: any) => updateConfig('lifecycle.garbage_collector.low_watermark', v)} className="w-24" />
        </SettingRow>
        <SettingRow label="启用定时调度" description="按照计划自动运行垃圾回收。">
          <Toggle checked={config.lifecycle.garbage_collector.enable_schedule} onChange={(v) => updateConfig('lifecycle.garbage_collector.enable_schedule', v)} />
        </SettingRow>
        <SettingRow label="间隔 (小时)" description="计划运行垃圾回收之间的时间。">
          <Input type="number" value={config.lifecycle.garbage_collector.interval_hours} onChange={(v: any) => updateConfig('lifecycle.garbage_collector.interval_hours', v)} className="w-24" />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
