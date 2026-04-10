import { SettingSection, SettingRow, Input, Select } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function InfrastructureSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="Qdrant (向量数据库)">
        <SettingRow label="主机" description="Qdrant 服务器的主机名或 IP。">
          <Input value={config.qdrant.host} onChange={(v: string) => updateConfig('qdrant.host', v)} />
        </SettingRow>
        <SettingRow label="端口" description="REST API 端口。">
          <Input type="number" value={config.qdrant.port} onChange={(v: number) => updateConfig('qdrant.port', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="集合名称" description="记忆向量的主集合名称。">
          <Input value={config.qdrant.collection_name} onChange={(v: string) => updateConfig('qdrant.collection_name', v)} className="w-64" />
        </SettingRow>
        <SettingRow label="距离度量" description="用于向量相似度计算的度量标准。">
          <Select 
            value={config.qdrant.distance_metric}
            onChange={(v: string) => updateConfig('qdrant.distance_metric', v)}
            error={getFieldError('qdrant.distance_metric')}
            options={[
              {label: '余弦相似度 (Cosine)', value: 'Cosine'},
              {label: '欧几里得距离 (Euclidean)', value: 'Euclidean'},
              {label: '点积 (Dot)', value: 'Dot'}
            ]} 
          />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
