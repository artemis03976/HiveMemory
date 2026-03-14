import React from 'react';
import { Save, RotateCcw, AlertCircle } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { CategorySection } from './settings/CategorySection';
import {
  TextInput,
  NumberInput,
  PasswordInput,
  ToggleSwitch,
  SelectDropdown,
  SliderInput,
  FilePathInput,
} from './settings/FormControls';
import { GatewayCategories } from './settings/GatewayCategories';
import { PerceptionCategories } from './settings/PerceptionCategories';
import { GenerationCategories } from './settings/GenerationCategories';
import { RetrievalCategories } from './settings/RetrievalCategories';
import { LifecycleCategories } from './settings/LifecycleCategories';
import { KoakumaCategories } from './settings/KoakumaCategories';

export const SettingsPanel: React.FC = () => {
  const {
    config,
    loading,
    validationErrors,
    isDirty,
    updateConfig,
    saveConfig,
    resetConfig,
  } = useSettings();

  const handleSave = async () => {
    try {
      await saveConfig();
      // Show success notification
      alert('配置保存成功');
    } catch (err) {
      alert('保存配置失败: ' + (err instanceof Error ? err.message : '未知错误'));
    }
  };

  const getFieldError = (field: string) => {
    return validationErrors.find((e) => e.field === field)?.message;
  };

  if (loading || !config) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-foreground/60">正在加载配置...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-background/50 backdrop-blur-lg">
        <div>
          <h2 className="text-xl font-bold text-foreground">系统设置</h2>
          <p className="text-sm text-muted-foreground mt-0.5">配置 HiveMemory 参数</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={resetConfig}
            disabled={!isDirty}
            className="px-4 py-2 bg-muted/20 hover:bg-muted/30 border border-white/10 rounded-lg text-sm transition-all duration-300 backdrop-blur-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </button>
          <button
            onClick={handleSave}
            disabled={!isDirty || validationErrors.filter((e) => e.severity === 'error').length > 0}
            className="px-4 py-2 bg-primary/20 hover:bg-primary/30 border border-primary/30 rounded-lg text-sm transition-all duration-300 backdrop-blur-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            保存更改
          </button>
        </div>
      </div>

      {/* Validation Errors Banner */}
      {validationErrors.filter((e) => e.severity === 'error').length > 0 && (
        <div className="mx-6 mt-4 px-4 py-3 bg-destructive/20 border border-destructive/30 rounded-lg flex items-start gap-3 backdrop-blur-lg">
          <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-destructive">配置存在验证错误</p>
            <p className="text-xs text-destructive/80 mt-1">
              请在保存前修复以下错误
            </p>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-6 py-4">
        <div className="max-w-5xl mx-auto space-y-4">
          {/* Basic Settings */}
          <CategorySection
            title="基础设置"
            paramCount={7}
            accentColor="hsl(var(--foreground) / 0.8)"
            defaultExpanded={true}
          >
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-foreground/90">系统</h4>
              <TextInput
                label="系统名称"
                value={config.system.name}
                onChange={(v) => updateConfig('system.name', v)}
                readOnly
                hint="只读系统标识符"
              />
              <TextInput
                label="版本"
                value={config.system.version}
                onChange={(v) => updateConfig('system.version', v)}
                readOnly
                hint="只读系统版本"
              />
              <ToggleSwitch
                label="调试模式"
                value={config.system.debug}
                onChange={(v) => updateConfig('system.debug', v)}
                hint="启用调试日志和详细输出"
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">日志</h4>
              <SelectDropdown
                label="日志级别"
                value={config.logging.level}
                onChange={(v) => updateConfig('logging.level', v)}
                options={[
                  { value: 'DEBUG', label: 'DEBUG' },
                  { value: 'INFO', label: 'INFO' },
                  { value: 'WARNING', label: 'WARNING' },
                  { value: 'ERROR', label: 'ERROR' },
                ]}
                error={getFieldError('logging.level')}
              />
              <TextInput
                label="日志格式"
                value={config.logging.format}
                onChange={(v) => updateConfig('logging.format', v)}
                hint="Python 日志格式字符串"
              />
              <FilePathInput
                label="日志文件路径"
                value={config.logging.file_path}
                onChange={(v) => updateConfig('logging.file_path', v)}
                placeholder="留空以禁用文件日志"
              />
              <ToggleSwitch
                label="控制台输出"
                value={config.logging.console_output}
                onChange={(v) => updateConfig('logging.console_output', v)}
                hint="启用控制台日志"
              />
            </div>
          </CategorySection>

          {/* Model Configuration */}
          <CategorySection
            title="模型配置"
            paramCount={18}
            accentColor="hsl(260, 80%, 65%)"
          >
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-foreground/90">网关 LLM</h4>
              <TextInput
                label="提供商"
                value={config.llm.gateway.provider}
                onChange={(v) => updateConfig('llm.gateway.provider', v)}
              />
              <TextInput
                label="模型"
                value={config.llm.gateway.model}
                onChange={(v) => updateConfig('llm.gateway.model', v)}
              />
              <PasswordInput
                label="API 密钥"
                value={config.llm.gateway.api_key}
                onChange={(v) => updateConfig('llm.gateway.api_key', v)}
                placeholder="留空以使用环境变量"
              />
              <TextInput
                label="API 基础 URL"
                value={config.llm.gateway.api_base || ''}
                onChange={(v) => updateConfig('llm.gateway.api_base', v || null)}
                placeholder="可选的自定义 API 端点"
              />
              <SliderInput
                label="温度"
                value={config.llm.gateway.temperature}
                onChange={(v) => updateConfig('llm.gateway.temperature', v)}
                min={0}
                max={2}
                step={0.1}
                error={getFieldError('llm.gateway.temperature')}
              />
              <NumberInput
                label="最大 Token 数"
                value={config.llm.gateway.max_tokens}
                onChange={(v) => updateConfig('llm.gateway.max_tokens', v)}
                min={1}
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">图书管理员 LLM</h4>
              <TextInput
                label="提供商"
                value={config.llm.librarian.provider}
                onChange={(v) => updateConfig('llm.librarian.provider', v)}
              />
              <TextInput
                label="模型"
                value={config.llm.librarian.model}
                onChange={(v) => updateConfig('llm.librarian.model', v)}
              />
              <PasswordInput
                label="API 密钥"
                value={config.llm.librarian.api_key}
                onChange={(v) => updateConfig('llm.librarian.api_key', v)}
                placeholder="留空以使用环境变量"
              />
              <TextInput
                label="API 基础 URL"
                value={config.llm.librarian.api_base || ''}
                onChange={(v) => updateConfig('llm.librarian.api_base', v || null)}
                placeholder="可选的自定义 API 端点"
              />
              <SliderInput
                label="温度"
                value={config.llm.librarian.temperature}
                onChange={(v) => updateConfig('llm.librarian.temperature', v)}
                min={0}
                max={2}
                step={0.1}
                error={getFieldError('llm.librarian.temperature')}
              />
              <NumberInput
                label="最大 Token 数"
                value={config.llm.librarian.max_tokens}
                onChange={(v) => updateConfig('llm.librarian.max_tokens', v)}
                min={1}
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">工作者 LLM</h4>
              <TextInput
                label="提供商"
                value={config.llm.worker.provider}
                onChange={(v) => updateConfig('llm.worker.provider', v)}
              />
              <TextInput
                label="模型"
                value={config.llm.worker.model}
                onChange={(v) => updateConfig('llm.worker.model', v)}
              />
              <PasswordInput
                label="API 密钥"
                value={config.llm.worker.api_key}
                onChange={(v) => updateConfig('llm.worker.api_key', v)}
                placeholder="留空以使用环境变量"
              />
              <TextInput
                label="API 基础 URL"
                value={config.llm.worker.api_base || ''}
                onChange={(v) => updateConfig('llm.worker.api_base', v || null)}
                placeholder="可选的自定义 API 端点"
              />
              <SliderInput
                label="温度"
                value={config.llm.worker.temperature}
                onChange={(v) => updateConfig('llm.worker.temperature', v)}
                min={0}
                max={2}
                step={0.1}
                error={getFieldError('llm.worker.temperature')}
              />
              <NumberInput
                label="最大 Token 数"
                value={config.llm.worker.max_tokens}
                onChange={(v) => updateConfig('llm.worker.max_tokens', v)}
                min={1}
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">默认嵌入模型</h4>
              <TextInput
                label="模型名称"
                value={config.embedding.default.model_name}
                onChange={(v) => updateConfig('embedding.default.model_name', v)}
              />
              <SelectDropdown
                label="设备"
                value={config.embedding.default.device}
                onChange={(v) => updateConfig('embedding.default.device', v)}
                options={[
                  { value: 'cpu', label: 'CPU' },
                  { value: 'cuda', label: 'CUDA (GPU)' },
                  { value: 'mps', label: 'MPS (Apple Silicon)' },
                ]}
                error={getFieldError('embedding.default.device')}
              />
              <FilePathInput
                label="缓存目录"
                value={config.embedding.default.cache_dir}
                onChange={(v) => updateConfig('embedding.default.cache_dir', v)}
                placeholder="留空以使用默认缓存位置"
              />
              <NumberInput
                label="批处理大小"
                value={config.embedding.default.batch_size}
                onChange={(v) => updateConfig('embedding.default.batch_size', v)}
                min={1}
              />
              <ToggleSwitch
                label="归一化嵌入"
                value={config.embedding.default.normalize_embeddings}
                onChange={(v) => updateConfig('embedding.default.normalize_embeddings', v)}
              />
              <NumberInput
                label="维度"
                value={config.embedding.default.dimension}
                onChange={(v) => updateConfig('embedding.default.dimension', v)}
                readOnly
                hint="只读嵌入维度"
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">感知嵌入模型</h4>
              <TextInput
                label="模型名称"
                value={config.embedding.perception.model_name}
                onChange={(v) => updateConfig('embedding.perception.model_name', v)}
              />
              <SelectDropdown
                label="设备"
                value={config.embedding.perception.device}
                onChange={(v) => updateConfig('embedding.perception.device', v)}
                options={[
                  { value: 'cpu', label: 'CPU' },
                  { value: 'cuda', label: 'CUDA (GPU)' },
                  { value: 'mps', label: 'MPS (Apple Silicon)' },
                ]}
                error={getFieldError('embedding.perception.device')}
              />
              <FilePathInput
                label="缓存目录"
                value={config.embedding.perception.cache_dir}
                onChange={(v) => updateConfig('embedding.perception.cache_dir', v)}
                placeholder="留空以使用默认缓存位置"
              />
              <NumberInput
                label="批处理大小"
                value={config.embedding.perception.batch_size}
                onChange={(v) => updateConfig('embedding.perception.batch_size', v)}
                min={1}
              />
              <ToggleSwitch
                label="归一化嵌入"
                value={config.embedding.perception.normalize_embeddings}
                onChange={(v) => updateConfig('embedding.perception.normalize_embeddings', v)}
              />
              <NumberInput
                label="维度"
                value={config.embedding.perception.dimension}
                onChange={(v) => updateConfig('embedding.perception.dimension', v)}
                readOnly
                hint="只读嵌入维度"
              />
            </div>
          </CategorySection>

          {/* Infrastructure */}
          <CategorySection
            title="基础设施"
            paramCount={13}
            accentColor="hsl(190, 90%, 50%)"
          >
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-foreground/90">Qdrant 向量数据库</h4>
              <TextInput
                label="主机"
                value={config.qdrant.host}
                onChange={(v) => updateConfig('qdrant.host', v)}
              />
              <NumberInput
                label="端口"
                value={config.qdrant.port}
                onChange={(v) => updateConfig('qdrant.port', v)}
                min={1}
                max={65535}
              />
              <NumberInput
                label="gRPC 端口"
                value={config.qdrant.grpc_port}
                onChange={(v) => updateConfig('qdrant.grpc_port', v)}
                min={1}
                max={65535}
              />
              <PasswordInput
                label="API 密钥"
                value={config.qdrant.api_key}
                onChange={(v) => updateConfig('qdrant.api_key', v)}
                placeholder="留空以禁用认证"
              />
              <TextInput
                label="集合名称"
                value={config.qdrant.collection_name}
                onChange={(v) => updateConfig('qdrant.collection_name', v)}
              />
              <NumberInput
                label="向量维度"
                value={config.qdrant.vector_dimension}
                onChange={(v) => updateConfig('qdrant.vector_dimension', v)}
                readOnly
                hint="必须与嵌入维度匹配"
              />
              <SelectDropdown
                label="距离度量"
                value={config.qdrant.distance_metric}
                onChange={(v) => updateConfig('qdrant.distance_metric', v)}
                options={[
                  { value: 'Cosine', label: '余弦相似度' },
                  { value: 'Euclidean', label: '欧几里得距离' },
                  { value: 'Dot', label: '点积' },
                ]}
                error={getFieldError('qdrant.distance_metric')}
              />
              <ToggleSwitch
                label="磁盘 Payload"
                value={config.qdrant.on_disk_payload}
                onChange={(v) => updateConfig('qdrant.on_disk_payload', v)}
                hint="将 Payload 存储在磁盘以节省内存"
              />

              <h4 className="text-sm font-semibold text-foreground/90 mt-6">Redis 缓存</h4>
              <TextInput
                label="主机"
                value={config.redis.host}
                onChange={(v) => updateConfig('redis.host', v)}
              />
              <NumberInput
                label="端口"
                value={config.redis.port}
                onChange={(v) => updateConfig('redis.port', v)}
                min={1}
                max={65535}
              />
              <PasswordInput
                label="密码"
                value={config.redis.password}
                onChange={(v) => updateConfig('redis.password', v)}
                placeholder="留空以禁用认证"
              />
              <NumberInput
                label="数据库"
                value={config.redis.db}
                onChange={(v) => updateConfig('redis.db', v)}
                min={0}
                max={15}
                error={getFieldError('redis.db')}
                hint="Redis 数据库索引 (0-15)"
              />
              <ToggleSwitch
                label="解码响应"
                value={config.redis.decode_responses}
                onChange={(v) => updateConfig('redis.decode_responses', v)}
                hint="自动将字节响应解码为字符串"
              />
            </div>
          </CategorySection>

          {/* Gateway Settings */}
          <GatewayCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />

          {/* Perception Layer */}
          <PerceptionCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />

          {/* Memory Generation */}
          <GenerationCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />

          {/* Memory Retrieval */}
          <RetrievalCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />

          {/* Lifecycle Management */}
          <LifecycleCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />

          {/* Advanced Settings */}
          <KoakumaCategories
            config={config}
            updateConfig={updateConfig}
            getFieldError={getFieldError}
          />
        </div>
      </div>
    </div>
  );
};
