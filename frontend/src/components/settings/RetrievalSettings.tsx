import { SettingSection, SettingRow, Input, Toggle, Select } from '../common/FormControls';

export function RetrievalSettings({ config, updateConfig, getFieldError }: any) {
  if (!config) return null;
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="记忆检索">
        <SettingRow label="渲染器类型" description="注入上下文的检索记忆格式。">
          <Select 
            value={config.retrieval.renderer.type}
            onChange={(v) => updateConfig('retrieval.renderer.type', v)}
            error={getFieldError('retrieval.renderer.type')}
            options={[
              {label: '完整 (Full)', value: 'full'},
              {label: '级联 (Cascade)', value: 'cascade'},
              {label: '紧凑 (Compact)', value: 'compact'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="检索器类型" description="记忆检索的搜索算法。">
          <Select 
            value={config.retrieval.retriever.type}
            onChange={(v) => updateConfig('retrieval.retriever.type', v)}
            error={getFieldError('retrieval.retriever.type')}
            options={[
              {label: '混合 (Hybrid)', value: 'hybrid'},
              {label: '仅稠密 (Dense Only)', value: 'dense'},
              {label: '仅稀疏 (Sparse Only)', value: 'sparse'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="Top K" description="要检索的记忆数量。">
          <Input type="number" value={config.retrieval.retriever.top_k} onChange={(v: any) => updateConfig('retrieval.retriever.top_k', v)} className="w-24" />
        </SettingRow>
        <SettingRow label="启用重排器 (Reranker)" description="使用交叉编码器对检索结果重新排序。">
          <Toggle checked={config.retrieval.retriever.reranker.enabled} onChange={(v) => updateConfig('retrieval.retriever.reranker.enabled', v)} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
