import { SettingSection, SettingRow, Input, Select, Toggle } from '../common/FormControls';

export function InfrastructureSettings({ config, updateConfig, getFieldError }: any) {
  if (!config) return null;
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="Qdrant (向量数据库)">
        <SettingRow label="主机" description="Qdrant 服务器的主机名或 IP。">
          <Input value={config.qdrant.host} onChange={(v: any) => updateConfig('qdrant.host', v)} />
        </SettingRow>
        <SettingRow label="端口" description="REST API 端口。">
          <Input type="number" value={config.qdrant.port} onChange={(v: any) => updateConfig('qdrant.port', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="集合名称" description="记忆向量的主集合名称。">
          <Input value={config.qdrant.collection_name} onChange={(v: any) => updateConfig('qdrant.collection_name', v)} className="w-64" />
        </SettingRow>
        <SettingRow label="距离度量" description="用于向量相似度计算的度量标准。">
          <Select 
            value={config.qdrant.distance_metric}
            onChange={(v) => updateConfig('qdrant.distance_metric', v)}
            error={getFieldError('qdrant.distance_metric')}
            options={[
              {label: '余弦相似度 (Cosine)', value: 'Cosine'},
              {label: '欧几里得距离 (Euclidean)', value: 'Euclidean'},
              {label: '点积 (Dot)', value: 'Dot'}
            ]} 
          />
        </SettingRow>
      </SettingSection>

      <SettingSection title="Redis">
        <SettingRow label="主机" description="Redis 服务器的主机名或 IP。">
          <Input value={config.redis.host} onChange={(v: any) => updateConfig('redis.host', v)} />
        </SettingRow>
        <SettingRow label="端口" description="Redis 服务器端口。">
          <Input type="number" value={config.redis.port} onChange={(v: any) => updateConfig('redis.port', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="数据库索引" description="Redis 逻辑数据库编号。">
          <Input type="number" value={config.redis.db} onChange={(v: any) => updateConfig('redis.db', v)} className="w-24" error={getFieldError('redis.db')} />
        </SettingRow>
        <SettingRow label="解码响应" description="自动将字节响应解码为字符串。">
          <Toggle checked={config.redis.decode_responses} onChange={(v) => updateConfig('redis.decode_responses', v)} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
