/**
 * Config API Client
 */
import { apiErrorMessage } from '@/services/apiError';

import type { ConfigResponse, HiveMemoryConfig } from '@/types/config';
import { apiUrl } from '@/transports/backendUrl';

export async function fetchConfig(): Promise<ConfigResponse> {
  const res = await fetch(apiUrl('/api/v1/config'));
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchConfig failed'));
  return res.json();
}

export async function updateConfig(config: HiveMemoryConfig): Promise<ConfigResponse> {
  const res = await fetch(apiUrl('/api/v1/config'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'updateConfig failed'));
  return res.json();
}

export async function fetchDefaultConfig(): Promise<HiveMemoryConfig> {
  const res = await fetch(apiUrl('/api/v1/config/defaults'));
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchDefaultConfig failed'));
  return res.json();
}
