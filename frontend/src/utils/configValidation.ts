import type { HiveMemoryConfig, ValidationError } from '../types/config';

export const validateConfig = (config: HiveMemoryConfig): ValidationError[] => {
  const errors: ValidationError[] = [];

  // Validate logging level
  const validLogLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
  if (!validLogLevels.includes(config.logging.level)) {
    errors.push({
      field: 'logging.level',
      message: `Log level must be one of: ${validLogLevels.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate LLM temperatures (0-2)
  const validateTemperature = (temp: number, field: string) => {
    if (temp < 0 || temp > 2) {
      errors.push({
        field,
        message: 'Temperature must be between 0 and 2',
        severity: 'error',
      });
    }
  };

  validateTemperature(config.llm.gateway.temperature, 'llm.gateway.temperature');
  validateTemperature(config.llm.librarian.temperature, 'llm.librarian.temperature');
  validateTemperature(config.llm.worker.temperature, 'llm.worker.temperature');

  // Validate embedding device
  const validDevices = ['cpu', 'cuda', 'mps'];
  if (!validDevices.includes(config.embedding.default.device)) {
    errors.push({
      field: 'embedding.default.device',
      message: `Device must be one of: ${validDevices.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate Qdrant distance metric
  const validMetrics = ['Cosine', 'Euclidean', 'Dot'];
  if (!validMetrics.includes(config.qdrant.distance_metric)) {
    errors.push({
      field: 'qdrant.distance_metric',
      message: `Distance metric must be one of: ${validMetrics.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate gateway analyzer context window (1-10)
  if (config.gateway.analyzer.context_window < 1 || config.gateway.analyzer.context_window > 10) {
    errors.push({
      field: 'gateway.analyzer.context_window',
      message: 'Context window must be between 1 and 10',
      severity: 'error',
    });
  }

  // Validate semantic threshold ranges (0-1)
  const validateThreshold = (value: number, field: string) => {
    if (value < 0 || value > 1) {
      errors.push({
        field,
        message: 'Threshold must be between 0 and 1',
        severity: 'error',
      });
    }
  };

  // Validate deduplicator thresholds (0-1)
  validateThreshold(config.generation.deduplicator.high_similarity_threshold, 'generation.deduplicator.high_similarity_threshold');
  validateThreshold(config.generation.deduplicator.low_similarity_threshold, 'generation.deduplicator.low_similarity_threshold');
  validateThreshold(config.generation.deduplicator.content_similarity_threshold, 'generation.deduplicator.content_similarity_threshold');

  // Validate retrieval renderer type
  const validRendererTypes = ['full', 'cascade', 'compact'];
  if (!validRendererTypes.includes(config.retrieval.renderer.type)) {
    errors.push({
      field: 'retrieval.renderer.type',
      message: `Renderer type must be one of: ${validRendererTypes.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate retrieval renderer format
  const validFormats = ['xml', 'markdown'];
  if (!validFormats.includes(config.retrieval.renderer.render_format)) {
    errors.push({
      field: 'retrieval.renderer.render_format',
      message: `Render format must be one of: ${validFormats.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate retriever type
  const validRetrieverTypes = ['hybrid', 'dense', 'sparse'];
  if (!validRetrieverTypes.includes(config.retrieval.retriever.type)) {
    errors.push({
      field: 'retrieval.retriever.type',
      message: `Retriever type must be one of: ${validRetrieverTypes.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate retriever score thresholds (0-1)
  validateThreshold(config.retrieval.retriever.score_threshold, 'retrieval.retriever.score_threshold');
  validateThreshold(config.retrieval.retriever.dense.score_threshold, 'retrieval.retriever.dense.score_threshold');
  validateThreshold(config.retrieval.retriever.sparse.score_threshold, 'retrieval.retriever.sparse.score_threshold');

  // Validate fusion type
  const validFusionTypes = ['rrf', 'adaptive'];
  if (!validFusionTypes.includes(config.retrieval.retriever.fusion.type)) {
    errors.push({
      field: 'retrieval.retriever.fusion.type',
      message: `Fusion type must be one of: ${validFusionTypes.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate fusion weights (0-2)
  const validateWeight = (value: number, field: string) => {
    if (value < 0 || value > 2) {
      errors.push({
        field,
        message: 'Weight must be between 0 and 2',
        severity: 'error',
      });
    }
  };

  validateWeight(config.retrieval.retriever.fusion.dense_weight, 'retrieval.retriever.fusion.dense_weight');
  validateWeight(config.retrieval.retriever.fusion.sparse_weight, 'retrieval.retriever.fusion.sparse_weight');

  // Validate reranker device
  const validRerankerDevices = ['cpu', 'cuda'];
  if (!validRerankerDevices.includes(config.retrieval.retriever.reranker.device)) {
    errors.push({
      field: 'retrieval.retriever.reranker.device',
      message: `Reranker device must be one of: ${validRerankerDevices.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate vitality calculator weights (0-2)
  const vitality = config.lifecycle.vitality_calculator;
  validateWeight(vitality.code_snippet_weight, 'lifecycle.vitality_calculator.code_snippet_weight');
  validateWeight(vitality.fact_weight, 'lifecycle.vitality_calculator.fact_weight');
  validateWeight(vitality.url_resource_weight, 'lifecycle.vitality_calculator.url_resource_weight');
  validateWeight(vitality.reflection_weight, 'lifecycle.vitality_calculator.reflection_weight');
  validateWeight(vitality.user_profile_weight, 'lifecycle.vitality_calculator.user_profile_weight');
  validateWeight(vitality.work_in_progress_weight, 'lifecycle.vitality_calculator.work_in_progress_weight');
  validateWeight(vitality.default_weight, 'lifecycle.vitality_calculator.default_weight');

  // Validate decay_lambda (0-0.1)
  if (vitality.decay_lambda < 0 || vitality.decay_lambda > 0.1) {
    errors.push({
      field: 'lifecycle.vitality_calculator.decay_lambda',
      message: 'Decay lambda must be between 0 and 0.1',
      severity: 'error',
    });
  }

  // Validate reinforcement negative_confidence_multiplier (0-1)
  validateThreshold(config.lifecycle.reinforcement.negative_confidence_multiplier, 'lifecycle.reinforcement.negative_confidence_multiplier');

  // Validate relay engine type
  const validRelayTypes = ['simple', 'llm'];
  if (!validRelayTypes.includes(config.perception.engine.relay.engine.type)) {
    errors.push({
      field: 'perception.engine.relay.engine.type',
      message: `Relay engine type must be one of: ${validRelayTypes.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate Koakuma MTP prompt language
  const validLanguages = ['zh', 'en'];
  if (!validLanguages.includes(config.koakuma.mtp_prompt.language)) {
    errors.push({
      field: 'koakuma.mtp_prompt.language',
      message: `Language must be one of: ${validLanguages.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate Koakuma MTP prompt role
  const validRoles = ['coder', 'chat', 'default'];
  if (!validRoles.includes(config.koakuma.mtp_prompt.role)) {
    errors.push({
      field: 'koakuma.mtp_prompt.role',
      message: `Role must be one of: ${validRoles.join(', ')}`,
      severity: 'error',
    });
  }

  return errors;
};

export const validateField = (
  field: string,
  _value: unknown,
  config: HiveMemoryConfig
): ValidationError | null => {
  const errors = validateConfig(config);
  return errors.find((e) => e.field === field) || null;
};
