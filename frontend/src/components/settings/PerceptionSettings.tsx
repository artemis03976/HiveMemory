import { SettingSection, SettingRow, Input } from '../common/FormControls';

export function PerceptionSettings({ config, updateConfig, getFieldError }: any) {
  if (!config) return null;
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="感知引擎 (Perception Engine)">
        <SettingRow label="引擎类型" description="用于语义流感知的算法模型。">
          <Input value={config.perception.engine.type} disabled className="opacity-50" />
        </SettingRow>
        <SettingRow label="空闲超时 (秒)" description="非活跃话题被折叠归档前的等待时间。">
          <Input type="number" value={config.perception.engine.idle_timeout_seconds} onChange={(v: any) => updateConfig('perception.engine.idle_timeout_seconds', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="扫描间隔 (秒)" description="后台感知扫描的频率。">
          <Input type="number" value={config.perception.engine.scan_interval_seconds} onChange={(v: any) => updateConfig('perception.engine.scan_interval_seconds', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="折叠 Token 阈值" description="触发话题内容折叠的高水位线阈值。">
          <Input type="number" value={config.perception.engine.fold_token_threshold} onChange={(v: any) => updateConfig('perception.engine.fold_token_threshold', v)} className="w-32" />
        </SettingRow>
      </SettingSection>

      <SettingSection title="语义吸附器 (Semantic Adsorber)">
        <SettingRow label="高语义阈值" description="强语义绑定的阈值 (直接吸附)。">
          <Input type="number" value={config.perception.engine.adsorber.semantic_threshold_high} onChange={(v: any) => updateConfig('perception.engine.adsorber.semantic_threshold_high', v)} step="0.01" className="w-24" error={getFieldError('perception.engine.adsorber.semantic_threshold_high')} />
        </SettingRow>
        <SettingRow label="低语义阈值" description="弱语义绑定的阈值 (进入候选区)。">
          <Input type="number" value={config.perception.engine.adsorber.semantic_threshold_low} onChange={(v: any) => updateConfig('perception.engine.adsorber.semantic_threshold_low', v)} step="0.01" className="w-24" error={getFieldError('perception.engine.adsorber.semantic_threshold_low')} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
