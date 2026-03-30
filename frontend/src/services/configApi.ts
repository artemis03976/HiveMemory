/**
 * Config API Client
 */

import type { HiveMemoryConfig } from '@/types/config';

export async function fetchConfig(): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config');
  if (!res.ok) throw new Error(`fetchConfig failed: ${res.status}`);
  return res.json();
}

export async function updateConfig(config: HiveMemoryConfig): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`updateConfig failed: ${res.status}`);
  return res.json();
}

export async function fetchDefaultConfig(): Promise<HiveMemoryConfig> {
  const res = await fetch('/api/v1/config/defaults');
  if (!res.ok) throw new Error(`fetchDefaultConfig failed: ${res.status}`);
  return res.json();
}

export async function fetchFoldTokenThreshold(): Promise<number> {
  const config = await fetchConfig();
  return config.perception.engine.fold_token_threshold;
}
