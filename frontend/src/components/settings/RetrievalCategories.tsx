import React from 'react';
import { CategorySection } from './CategorySection';
import {
  NumberInput,
  ToggleSwitch,
  SelectDropdown,
  SliderInput,
  TextInput,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface RetrievalCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const RetrievalCategories: React.FC<RetrievalCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="记忆检索"
      paramCount={27}
      accentColor="hsl(30, 20%, 50%)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">渲染器</h4>
        <SelectDropdown
          label="类型"
          value={config.retrieval.renderer.type}
          onChange={(v) => updateConfig('retrieval.renderer.type', v)}
          options={[
            { value: 'full', label: '完整' },
            { value: 'cascade', label: '级联' },
            { value: 'compact', label: '紧凑' },
          ]}
          error={getFieldError('retrieval.renderer.type')}
        />
        <SelectDropdown
          label="渲染格式"
          value={config.retrieval.renderer.render_format}
          onChange={(v) => updateConfig('retrieval.renderer.render_format', v)}
          options={[
            { value: 'xml', label: 'XML' },
            { value: 'markdown', label: 'Markdown' },
          ]}
          error={getFieldError('retrieval.renderer.render_format')}
        />
        <NumberInput
          label="最大 Token 数"
          value={config.retrieval.renderer.max_tokens}
          onChange={(v) => updateConfig('retrieval.renderer.max_tokens', v)}
          min={0}
        />
        <NumberInput
          label="最大内容长度"
          value={config.retrieval.renderer.max_content_length}
          onChange={(v) => updateConfig('retrieval.renderer.max_content_length', v)}
          min={0}
        />
        <ToggleSwitch
          label="显示产物"
          value={config.retrieval.renderer.show_artifacts}
          onChange={(v) => updateConfig('retrieval.renderer.show_artifacts', v)}
        />
        <NumberInput
          label="过期天数"
          value={config.retrieval.renderer.stale_days}
          onChange={(v) => updateConfig('retrieval.renderer.stale_days', v)}
          min={0}
          hint="记忆被认为过期前的天数"
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">检索器</h4>
        <SelectDropdown
          label="类型"
          value={config.retrieval.retriever.type}
          onChange={(v) => updateConfig('retrieval.retriever.type', v)}
          options={[
            { value: 'hybrid', label: '混合' },
            { value: 'dense', label: '仅稠密' },
            { value: 'sparse', label: '仅稀疏' },
          ]}
          error={getFieldError('retrieval.retriever.type')}
        />
        <NumberInput
          label="Top K"
          value={config.retrieval.retriever.top_k}
          onChange={(v) => updateConfig('retrieval.retriever.top_k', v)}
          min={1}
        />
        <SliderInput
          label="分数阈值"
          value={config.retrieval.retriever.score_threshold}
          onChange={(v) => updateConfig('retrieval.retriever.score_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('retrieval.retriever.score_threshold')}
        />
        <ToggleSwitch
          label="启用并行"
          value={config.retrieval.retriever.enable_parallel}
          onChange={(v) => updateConfig('retrieval.retriever.enable_parallel', v)}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">稠密检索</h4>
        <ToggleSwitch
          label="启用"
          value={config.retrieval.retriever.dense.enabled}
          onChange={(v) => updateConfig('retrieval.retriever.dense.enabled', v)}
        />
        <NumberInput
          label="Top K"
          value={config.retrieval.retriever.dense.top_k}
          onChange={(v) => updateConfig('retrieval.retriever.dense.top_k', v)}
          min={1}
        />
        <SliderInput
          label="分数阈值"
          value={config.retrieval.retriever.dense.score_threshold}
          onChange={(v) => updateConfig('retrieval.retriever.dense.score_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('retrieval.retriever.dense.score_threshold')}
        />
        <ToggleSwitch
          label="启用时间衰减"
          value={config.retrieval.retriever.dense.enable_time_decay}
          onChange={(v) => updateConfig('retrieval.retriever.dense.enable_time_decay', v)}
        />
        <NumberInput
          label="时间衰减天数"
          value={config.retrieval.retriever.dense.time_decay_days}
          onChange={(v) => updateConfig('retrieval.retriever.dense.time_decay_days', v)}
          min={1}
        />
        <ToggleSwitch
          label="启用置信度提升"
          value={config.retrieval.retriever.dense.enable_confidence_boost}
          onChange={(v) => updateConfig('retrieval.retriever.dense.enable_confidence_boost', v)}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">稀疏检索</h4>
        <ToggleSwitch
          label="启用"
          value={config.retrieval.retriever.sparse.enabled}
          onChange={(v) => updateConfig('retrieval.retriever.sparse.enabled', v)}
        />
        <NumberInput
          label="Top K"
          value={config.retrieval.retriever.sparse.top_k}
          onChange={(v) => updateConfig('retrieval.retriever.sparse.top_k', v)}
          min={1}
        />
        <SliderInput
          label="分数阈值"
          value={config.retrieval.retriever.sparse.score_threshold}
          onChange={(v) => updateConfig('retrieval.retriever.sparse.score_threshold', v)}
          min={0}
          max={1}
          step={0.01}
          error={getFieldError('retrieval.retriever.sparse.score_threshold')}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">融合</h4>
        <SelectDropdown
          label="类型"
          value={config.retrieval.retriever.fusion.type}
          onChange={(v) => updateConfig('retrieval.retriever.fusion.type', v)}
          options={[
            { value: 'rrf', label: '倒数排名融合 (RRF)' },
            { value: 'adaptive', label: '自适应' },
          ]}
          error={getFieldError('retrieval.retriever.fusion.type')}
        />
        <NumberInput
          label="RRF K"
          value={config.retrieval.retriever.fusion.rrf_k}
          onChange={(v) => updateConfig('retrieval.retriever.fusion.rrf_k', v)}
          min={1}
        />
        <SliderInput
          label="稠密权重"
          value={config.retrieval.retriever.fusion.dense_weight}
          onChange={(v) => updateConfig('retrieval.retriever.fusion.dense_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('retrieval.retriever.fusion.dense_weight')}
        />
        <SliderInput
          label="稀疏权重"
          value={config.retrieval.retriever.fusion.sparse_weight}
          onChange={(v) => updateConfig('retrieval.retriever.fusion.sparse_weight', v)}
          min={0}
          max={2}
          step={0.1}
          error={getFieldError('retrieval.retriever.fusion.sparse_weight')}
        />
        <NumberInput
          label="最终 Top K"
          value={config.retrieval.retriever.fusion.final_top_k}
          onChange={(v) => updateConfig('retrieval.retriever.fusion.final_top_k', v)}
          min={1}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">重排序器</h4>
        <ToggleSwitch
          label="启用"
          value={config.retrieval.retriever.reranker.enabled}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.enabled', v)}
        />
        <TextInput
          label="模型名称"
          value={config.retrieval.retriever.reranker.model_name}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.model_name', v)}
        />
        <SelectDropdown
          label="设备"
          value={config.retrieval.retriever.reranker.device}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.device', v)}
          options={[
            { value: 'cpu', label: 'CPU' },
            { value: 'cuda', label: 'CUDA (GPU)' },
          ]}
          error={getFieldError('retrieval.retriever.reranker.device')}
        />
        <ToggleSwitch
          label="使用 FP16"
          value={config.retrieval.retriever.reranker.use_fp16}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.use_fp16', v)}
          hint="使用半精度以加快推理速度"
        />
        <NumberInput
          label="批处理大小"
          value={config.retrieval.retriever.reranker.batch_size}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.batch_size', v)}
          min={1}
        />
        <NumberInput
          label="Top K"
          value={config.retrieval.retriever.reranker.top_k}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.top_k', v)}
          min={1}
        />
        <ToggleSwitch
          label="归一化分数"
          value={config.retrieval.retriever.reranker.normalize_scores}
          onChange={(v) => updateConfig('retrieval.retriever.reranker.normalize_scores', v)}
        />
      </div>
    </CategorySection>
  );
};
