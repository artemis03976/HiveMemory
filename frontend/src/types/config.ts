// Configuration types matching config.yaml structure

export interface SystemConfig {
  name: string;
  version: string;
  debug: boolean;
}

export interface LoggingConfig {
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  format: string;
  file_path: string | null;
  console_output: boolean;
}

export interface LLMConfig {
  provider: string;
  model: string;
  api_key: string | null;
  api_base: string | null;
  temperature: number;
  max_tokens: number;
}

export interface LLMsConfig {
  gateway: LLMConfig;
  librarian: LLMConfig;
  worker: LLMConfig;
}

export interface EmbeddingConfig {
  model_name: string;
  device: 'cpu' | 'cuda' | 'mps';
  cache_dir: string | null;
  batch_size: number;
  normalize_embeddings: boolean;
  dimension: number;
}

export interface EmbeddingsConfig {
  default: EmbeddingConfig;
}

export interface QdrantConfig {
  host: string;
  port: number;
  grpc_port: number;
  api_key: string | null;
  collection_name: string;
  vector_dimension: number;
  distance_metric: 'Cosine' | 'Euclidean' | 'Dot';
  on_disk_payload: boolean;
}

export interface InterceptorConfig {
  enabled: boolean;
  enable_system: boolean;
  enable_chat: boolean;
  custom_system_patterns: string[];
  custom_chat_patterns: string[];
}

export interface AnalyzerConfig {
  enabled: boolean;
  context_window: number;
}

export interface GatewayConfig {
  interceptor: InterceptorConfig;
  analyzer: AnalyzerConfig;
}

export interface RelayEngineConfig {
  type: 'simple' | 'llm';
}

export interface RelayConfig {
  enable: boolean;
  engine: RelayEngineConfig;
}

export interface PerceptionEngineConfig {
  enable: boolean;
  idle_timeout_seconds: number;
  scan_interval_seconds: number;
  fold_token_threshold: number;
  fold_retain_recent_blocks: number;
  max_resident_topics: number;
  relay: RelayConfig;
}

export interface PerceptionConfig {
  engine: PerceptionEngineConfig;
}

export interface ExtractorConfig {
  enabled: boolean;
  system_prompt: string | null;
  user_prompt: string | null;
}

export interface DeduplicatorConfig {
  enabled: boolean;
  high_similarity_threshold: number;
  low_similarity_threshold: number;
  content_similarity_threshold: number;
  enable_vitality_tracking: boolean;
}

export interface GenerationConfig {
  extractor: ExtractorConfig;
  deduplicator: DeduplicatorConfig;
}

export interface RendererConfig {
  type: 'full' | 'cascade' | 'compact';
  render_format: 'xml' | 'markdown';
  max_tokens: number;
  max_content_length: number;
  show_artifacts: boolean;
  stale_days: number;
}

export interface DenseRetrieverConfig {
  enabled: boolean;
  top_k: number;
  score_threshold: number;
  enable_time_decay: boolean;
  time_decay_days: number;
  enable_confidence_boost: boolean;
}

export interface SparseRetrieverConfig {
  enabled: boolean;
  top_k: number;
  score_threshold: number;
}

export interface FusionConfig {
  type: 'rrf' | 'adaptive';
  rrf_k: number;
  dense_weight: number;
  sparse_weight: number;
  final_top_k: number;
}

export interface RerankerConfig {
  enabled: boolean;
  model_name: string;
  device: 'cpu' | 'cuda';
  cache_dir: string | null;
  use_fp16: boolean;
  batch_size: number;
  top_k: number;
  normalize_scores: boolean;
}

export interface RetrieverConfig {
  type: 'hybrid' | 'dense' | 'sparse';
  top_k: number;
  score_threshold: number;
  enable_parallel: boolean;
  dense: DenseRetrieverConfig;
  sparse: SparseRetrieverConfig;
  fusion: FusionConfig;
  reranker: RerankerConfig;
}

export interface RetrievalConfig {
  renderer: RendererConfig;
  retriever: RetrieverConfig;
}

export interface VitalityCalculatorConfig {
  code_snippet_weight: number;
  fact_weight: number;
  url_resource_weight: number;
  reflection_weight: number;
  user_profile_weight: number;
  work_in_progress_weight: number;
  default_weight: number;
  max_access_boost: number;
  points_per_access: number;
  decay_lambda: number;
}

export interface ReinforcementConfig {
  enable_event_history: boolean;
  event_history_limit: number;
  hit_boost: number;
  citation_boost: number;
  positive_feedback_boost: number;
  negative_feedback_penalty: number;
  negative_confidence_multiplier: number;
}

export interface ArchiverConfig {
  archive_dir: string;
  compression: boolean;
}

export interface GarbageCollectorConfig {
  low_watermark: number;
  batch_size: number;
  enable_schedule: boolean;
  interval_hours: number;
}

export interface LifecycleConfig {
  high_watermark: number;
  vitality_calculator: VitalityCalculatorConfig;
  reinforcement: ReinforcementConfig;
  archiver: ArchiverConfig;
  garbage_collector: GarbageCollectorConfig;
}

export interface MtpPromptConfig {
  enabled: boolean;
  language: 'zh' | 'en';
  role: 'coder' | 'chat' | 'default';
  include_demo: boolean;
  include_error_handling: boolean;
  include_kernel_tools: boolean;
}

export interface KoakumaConfig {
  enabled: boolean;
  execution_timeout_seconds: number;
  max_recursion_depth: number;
  tool_cache_size: number;
  python_repl_timeout_seconds: number;
  workspace_path: string;
  file_read_max_bytes: number;
  file_write_max_bytes: number;
  web_search_timeout_seconds: number;
  mtp_prompt: MtpPromptConfig;
}

export interface HiveMemoryConfig {
  system: SystemConfig;
  logging: LoggingConfig;
  llm: LLMsConfig;
  embedding: EmbeddingsConfig;
  qdrant: QdrantConfig;
  gateway: GatewayConfig;
  perception: PerceptionConfig;
  generation: GenerationConfig;
  retrieval: RetrievalConfig;
  lifecycle: LifecycleConfig;
  koakuma: KoakumaConfig;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}
