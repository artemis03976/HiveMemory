/**
 * Config API Client
 */
import { apiErrorMessage } from '@/services/apiError';

import type { HiveMemoryConfig } from '@/types/config';

export async function fetchConfig(): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config');
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchConfig failed'));
  return res.json();
}

export async function updateConfig(config: HiveMemoryConfig): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'updateConfig failed'));
  return res.json();
}

export async function fetchDefaultConfig(): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config/defaults');
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchDefaultConfig failed'));
  return res.json();
}

export async function fetchFoldTokenThreshold(): Promise<number> {
  const config = await fetchConfig();
  return config.perception.engine.fold_token_threshold;
}
