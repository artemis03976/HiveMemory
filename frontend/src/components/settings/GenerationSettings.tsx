import { SettingSection, SettingRow, Input, Toggle } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function GenerationSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="记忆生成">
        <SettingRow label="启用提取器" description="从对话中提取结构化的记忆原子。">
          <Toggle checked={config.generation.extractor.enabled} onChange={(v: boolean) => updateConfig('generation.extractor.enabled', v)} />
        </SettingRow>
        <SettingRow label="启用去重器" description="合并相似的记忆以防止重复。">
          <Toggle checked={config.generation.deduplicator.enabled} onChange={(v: boolean) => updateConfig('generation.deduplicator.enabled', v)} />
        </SettingRow>
        <SettingRow label="高相似度阈值" description="触发 TOUCH/UPDATE 操作的阈值。">
          <Input type="number" value={config.generation.deduplicator.high_similarity_threshold} onChange={(v: number) => updateConfig('generation.deduplicator.high_similarity_threshold', v)} step="0.01" className="w-24" error={getFieldError('generation.deduplicator.high_similarity_threshold')} />
        </SettingRow>
        <SettingRow label="低相似度阈值" description="触发 UPDATE/CREATE 操作的阈值。">
          <Input type="number" value={config.generation.deduplicator.low_similarity_threshold} onChange={(v: number) => updateConfig('generation.deduplicator.low_similarity_threshold', v)} step="0.01" className="w-24" error={getFieldError('generation.deduplicator.low_similarity_threshold')} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
