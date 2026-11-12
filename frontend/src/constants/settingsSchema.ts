/**
 * 设置页的两层：常用是日常会调整的项，高级是一般不需要改动的项。每一项绑定到主配置的一个
 * 字段路径；保存后何时生效由配置接口给出的 immediate_fields 决定，不在这里写死。
 */
export type SettingKind = 'number' | 'switch' | 'select' | 'model' | 'readonly';

export interface SettingField {
  path: string;
  label: string;
  description: string;
  kind: SettingKind;
  unit?: string;
  /** 显示值 = 配置值 × scale，例如字节显示为 MB、秒显示为分钟。 */
  scale?: number;
  step?: number;
  min?: number;
  options?: { label: string; value: string }[];
  /** 留空表示沿用模型定义等上层默认值，保存为 null。 */
  nullable?: boolean;
}

export interface SettingSectionDef {
  id: string;
  title: string;
  /** 风格化的名称，作为小节标题旁的标签。 */
  persona?: string;
  description: string;
  fields: SettingField[];
}

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'].map((v) => ({ label: v, value: v }));
const RESIDENCY = [
  { label: '空闲一段时间后卸载', value: 'idle_unload' },
  { label: '常驻内存', value: 'resident' },
];

export const COMMON_SECTIONS: SettingSectionDef[] = [
  {
    id: 'model',
    title: '模型',
    description: 'Gateway 负责入口分析与话题路由，帕秋莉负责记忆的检索、感知与生成',
    fields: [
      { path: 'gateway.llm.model_id', label: 'Gateway 使用的模型', description: '从模型注册表中选择', kind: 'model' },
      { path: 'patchouli.llm.model_id', label: '帕秋莉使用的模型', description: '从模型注册表中选择', kind: 'model' },
    ],
  },
  {
    id: 'memory',
    title: '记忆',
    persona: '帕秋莉',
    description: '检索多少条、是否自动提取、话题多久结算',
    fields: [
      { path: 'patchouli.retrieval.retriever.top_k', label: '检索条数上限', description: '每次检索最多返回几条记忆，至少为 1', kind: 'number', unit: '条', min: 1, step: 1 },
      { path: 'patchouli.generation.extractor.enabled', label: '自动从对话中提取记忆', description: '话题结算时，从对话里提炼出长期记忆', kind: 'switch' },
      { path: 'patchouli.perception.idle_timeout_seconds', label: '对话停多久后结算话题', description: '超过这段时间没有新消息，话题自动结算', kind: 'number', unit: '分钟', scale: 1 / 60, min: 1, step: 1 },
    ],
  },
  {
    id: 'chat',
    title: '对话',
    description: '人偶执行对话时的上下文处理',
    fields: [
      { path: 'alice.compaction.enabled', label: '自动压缩上下文', description: '历史接近模型窗口时，先把较早的部分压缩成摘要', kind: 'switch' },
    ],
  },
  {
    id: 'attachment',
    title: '附件',
    description: '',
    fields: [
      { path: 'attachment_parser.max_raw_bytes', label: '上传大小上限', description: '单个附件的最大体积', kind: 'number', unit: 'MB', scale: 1 / (1024 * 1024), min: 1, step: 1 },
    ],
  },
  {
    id: 'system',
    title: '系统',
    description: '',
    fields: [
      { path: 'logging.level', label: '日志级别', description: '排查问题时可调到 DEBUG', kind: 'select', options: LOG_LEVELS },
      { path: 'i18n.default_language', label: '默认语言', description: '提示词与系统消息使用的语言', kind: 'select', options: [{ label: '中文', value: 'zh' }, { label: 'English', value: 'en' }] },
    ],
  },
];

export const ADVANCED_SECTIONS: SettingSectionDef[] = [
  {
    id: 'a-model',
    title: '模型',
    description: '采样参数与本地模型的驻留；留空的参数沿用模型定义',
    fields: [
      { path: 'gateway.llm.temperature', label: 'Gateway 采样温度', description: '路由需要稳定，一般保持很低', kind: 'number', step: 0.1, min: 0, nullable: true },
      { path: 'gateway.llm.top_p', label: 'Gateway Top P', description: '留空沿用模型定义', kind: 'number', step: 0.05, min: 0, nullable: true },
      { path: 'gateway.llm.max_tokens', label: 'Gateway 最大输出', description: '留空沿用模型定义', kind: 'number', unit: 'tokens', step: 1, min: 1, nullable: true },
      { path: 'patchouli.llm.temperature', label: '帕秋莉采样温度', description: '留空沿用模型定义', kind: 'number', step: 0.1, min: 0, nullable: true },
      { path: 'patchouli.llm.top_p', label: '帕秋莉 Top P', description: '留空沿用模型定义', kind: 'number', step: 0.05, min: 0, nullable: true },
      { path: 'patchouli.llm.max_tokens', label: '帕秋莉最大输出', description: '留空沿用模型定义', kind: 'number', unit: 'tokens', step: 1, min: 1, nullable: true },
      { path: 'shared.embedding.default.residency', label: '嵌入模型的驻留方式', description: '常驻更快，空闲卸载更省内存', kind: 'select', options: RESIDENCY },
      { path: 'shared.embedding.default.idle_timeout_seconds', label: '嵌入模型空闲卸载时间', description: '空闲卸载时，多久没有使用就卸载', kind: 'number', unit: '分钟', scale: 1 / 60, min: 1, step: 1 },
      { path: 'patchouli.retrieval.reranker.residency', label: '重排模型的驻留方式', description: '常驻更快，空闲卸载更省内存', kind: 'select', options: RESIDENCY },
      { path: 'patchouli.retrieval.reranker.idle_timeout_seconds', label: '重排模型空闲卸载时间', description: '空闲卸载时，多久没有使用就卸载', kind: 'number', unit: '分钟', scale: 1 / 60, min: 1, step: 1 },
    ],
  },
  {
    id: 'a-retrieval',
    title: '检索',
    persona: '帕秋莉',
    description: '相关度阈值、时间衰减、重排与注入上下文的上限',
    fields: [
      { path: 'patchouli.retrieval.retriever.dense.score_threshold', label: '相关度阈值', description: '低于这个分数的结果不返回', kind: 'number', step: 0.05, min: 0 },
      { path: 'patchouli.retrieval.retriever.dense.enable_time_decay', label: '按时间衰减', description: '较旧的记忆排序靠后', kind: 'switch' },
      { path: 'patchouli.retrieval.retriever.dense.time_decay_days', label: '时间衰减的半程', description: '经过这么多天，时间因素的权重减半', kind: 'number', unit: '天', min: 1, step: 1 },
      { path: 'patchouli.retrieval.reranker.enabled', label: '重排', description: '用重排模型对召回结果重新排序', kind: 'switch' },
      { path: 'memory_compiler.retrieval_context.strategy.max_memory_tokens', label: '注入上下文的记忆上限', description: '检索到的记忆最多占用多少 tokens', kind: 'number', unit: 'tokens', min: 1, step: 100 },
    ],
  },
  {
    id: 'a-generation',
    title: '生成',
    persona: '帕秋莉',
    description: '查重与并发',
    fields: [
      { path: 'patchouli.generation.deduplicator.high_similarity_threshold', label: '查重：高相似阈值', description: '高于它直接视为同一条', kind: 'number', step: 0.01, min: 0 },
      { path: 'patchouli.generation.deduplicator.low_similarity_threshold', label: '查重：低相似阈值', description: '介于两个阈值之间时提名合并', kind: 'number', step: 0.01, min: 0 },
      { path: 'patchouli.generation.queue_max_concurrency', label: '并发提取数', description: '同时执行的生成任务数', kind: 'number', min: 1, step: 1 },
    ],
  },
  {
    id: 'a-lifecycle',
    title: '话题与生命周期',
    persona: '帕秋莉',
    description: '驻留、遗忘与归档',
    fields: [
      { path: 'patchouli.perception.engine.max_resident_topics', label: '同时驻留的话题数', description: '超过时最久没用的话题被结算', kind: 'number', min: 1, step: 1 },
      { path: 'patchouli.lifecycle.vitality_calculator.decay_lambda', label: '遗忘速度', description: '越大，长期不用的记忆生命力下降越快', kind: 'number', step: 0.001, min: 0 },
      { path: 'patchouli.lifecycle.garbage_collector.low_watermark', label: '低于多少生命力时归档', description: '自动归档的阈值', kind: 'number', min: 0, step: 1 },
      { path: 'scheduler.tasks.enable_lifecycle_gc', label: '自动归档', description: '按周期扫描并归档低生命力的记忆', kind: 'switch' },
      { path: 'scheduler.tasks.lifecycle_gc_interval_hours', label: '自动归档的周期', description: '多久扫描一次', kind: 'number', unit: '小时', min: 1, step: 1 },
    ],
  },
  {
    id: 'a-chat',
    title: '对话与附件',
    description: '压缩时机、工具调用轮数与上下文窗口',
    fields: [
      { path: 'alice.compaction.trigger_ratio', label: '压缩触发比例', description: '上下文用到窗口的这个比例时开始压缩', kind: 'number', step: 0.05, min: 0 },
      { path: 'alice.runtime.max_loop_iterations', label: '单轮最多工具调用轮数', description: '一轮对话中人偶最多连续调用几次工具', kind: 'number', min: 1, step: 1 },
      { path: 'alice.runtime.default_context_window', label: '默认上下文窗口', description: '模型定义没有给出窗口时使用', kind: 'number', unit: 'tokens', min: 1, step: 1024 },
      { path: 'attachment_compiler.max_total_context_chars', label: '注入上下文的附件上限', description: '一轮最多注入多少字符的附件内容', kind: 'number', unit: '字符', min: 1, step: 1000 },
    ],
  },
  {
    id: 'a-runtime',
    title: '运行时与存储',
    description: '请求时限、会话上限与日志文件',
    fields: [
      { path: 'gateway.workflow.default_request_timeout_ms', label: 'Gateway 请求时限', description: '一次入口分析最多等待多久', kind: 'number', unit: '毫秒', min: 1, step: 500 },
      { path: 'import_bus.apply_wait_seconds', label: '导入等待应用的时限', description: '外部 harness 回流的一轮最多等待多久应用', kind: 'number', unit: '秒', min: 1, step: 1 },
      { path: 'workspace.sessions.max_sessions_per_workspace', label: '会话数上限', description: '一个 Workspace 最多保留多少会话', kind: 'number', min: 1, step: 1 },
      { path: 'workspace.sessions.max_turns_per_session', label: '每个会话的轮数上限', description: '超过后需要新建会话', kind: 'number', min: 1, step: 1 },
      { path: 'runtime_events.enabled', label: '运行页事件流', description: '关闭后运行页看不到运行时事件', kind: 'switch' },
      { path: 'patchouli.storage.host', label: '存储地址', description: '由安装位置决定', kind: 'readonly' },
      { path: 'patchouli.storage.collection_name', label: '集合名', description: '由安装位置决定', kind: 'readonly' },
    ],
  },
];

/** 字段路径对应的名称，用于保存栏与等待重启的提示；不在两层中的字段显示路径本身。 */
export function settingLabel(path: string): string {
  for (const section of [...COMMON_SECTIONS, ...ADVANCED_SECTIONS]) {
    const field = section.fields.find((f) => f.path === path);
    if (field) return field.label;
  }
  return path;
}

export function readPath(config: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>(
    (node, key) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[key] : undefined),
    config,
  );
}

/** 两层中值与原配置不同的字段路径，保存栏据此列出未保存的更改。 */
export function changedSettingPaths(original: unknown, draft: unknown): string[] {
  return [...COMMON_SECTIONS, ...ADVANCED_SECTIONS]
    .flatMap((section) => section.fields)
    .filter((field) => JSON.stringify(readPath(original, field.path)) !== JSON.stringify(readPath(draft, field.path)))
    .map((field) => field.path);
}
