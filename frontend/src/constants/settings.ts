import type { HiveMemoryConfig } from '@/types/config';

/**
 * 调试模式下读不到后端配置时，设置页使用的配置。结构与后端主配置一致，
 * 只含前端读取的字段，取值为 configs/config.yaml 的默认值。
 */
export const MOCK_CONFIG: HiveMemoryConfig = {
  logging: { level: 'INFO' },
  i18n: { default_language: 'zh' },
  runtime_events: { enabled: true },
  scheduler: {
    tasks: { enable_lifecycle_gc: true, lifecycle_gc_interval_hours: 24 },
  },
  shared: {
    embedding: { default: { residency: 'idle_unload', idle_timeout_seconds: 1800 } },
  },
  gateway: {
    llm: { model_id: 'deepseek-chat', temperature: 0.0, top_p: null, max_tokens: 512 },
    workflow: { default_request_timeout_ms: 8000 },
  },
  import_bus: { apply_wait_seconds: 30 },
  memory_compiler: {
    retrieval_context: { strategy: { max_memory_tokens: 2000 } },
  },
  patchouli: {
    llm: { model_id: 'deepseek-chat', temperature: 0.3, top_p: null, max_tokens: 32768 },
    storage: { host: '127.0.0.1', collection_name: 'hivememory_main' },
    perception: {
      idle_timeout_seconds: 3600,
      engine: { max_resident_topics: 5, fold_token_threshold: 131072, fold_retain_recent_blocks: 2 },
    },
    generation: {
      extractor: { enabled: true },
      deduplicator: { high_similarity_threshold: 0.95, low_similarity_threshold: 0.75 },
      queue_max_concurrency: 2,
    },
    retrieval: {
      retriever: {
        top_k: 5,
        dense: { score_threshold: 0.5, enable_time_decay: true, time_decay_days: 30 },
      },
      reranker: { enabled: true, residency: 'idle_unload', idle_timeout_seconds: 600 },
    },
    lifecycle: {
      vitality_calculator: { decay_lambda: 0.01 },
      garbage_collector: { low_watermark: 20 },
    },
  },
  alice: {
    compaction: { enabled: true, trigger_ratio: 0.75 },
    runtime: { max_loop_iterations: 10, default_context_window: 65536 },
  },
  attachment_parser: { max_raw_bytes: 10485760 },
  attachment_compiler: { max_total_context_chars: 48000 },
  workspace: {
    sessions: { max_sessions_per_workspace: 128, max_turns_per_session: 512 },
  },
};
