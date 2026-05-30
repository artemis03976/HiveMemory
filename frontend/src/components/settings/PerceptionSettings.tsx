import { SettingSection, SettingRow, Input } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function PerceptionSettings({ config, updateConfig }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="感知引擎 (Perception Engine)">
        <SettingRow label="引擎类型" description="用于语义流感知的算法模型。">
          <Input value={config.perception.engine.type} disabled className="opacity-50" />
        </SettingRow>
        <SettingRow label="空闲超时 (秒)" description="非活跃话题被折叠归档前的等待时间。">
          <Input 
            type="number" 
            value={config.perception.engine.idle_timeout_seconds} 
            onChange={(v: number) => updateConfig('perception.engine.idle_timeout_seconds', v)} 
            className="w-32" 
          />
        </SettingRow>
        <SettingRow label="扫描间隔 (秒)" description="后台感知扫描的频率。">
          <Input 
            type="number" 
            value={config.perception.engine.scan_interval_seconds} 
            onChange={(v: number) => updateConfig('perception.engine.scan_interval_seconds', v)} 
            className="w-32" 
          />
        </SettingRow>
        <SettingRow label="折叠 Token 阈值" description="触发话题内容折叠的高水位线阈值。">
          <Input
            type="number"
            value={config.perception.engine.fold_token_threshold}
            onChange={(v: number) => updateConfig('perception.engine.fold_token_threshold', v)}
            className="w-32"
          />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
