import type { RegisteredProvider, ProviderUpsertPayload } from '@/types/provider';

const BASE = '/api/v1';

/** 获取所有已配置的提供商（API 密钥已脱敏） */
export async function fetchProviders(): Promise<RegisteredProvider[]> {
  const res = await fetch(`${BASE}/providers`);
  if (!res.ok) throw new Error(`获取提供商列表失败: ${res.status}`);
  return res.json();
}

/** 创建或更新提供商凭证 */
export async function upsertProvider(
  name: string,
  payload: ProviderUpsertPayload,
): Promise<RegisteredProvider> {
  const res = await fetch(`${BASE}/providers/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `保存提供商失败: ${res.status}`);
  }
  return res.json();
}

/** 删除提供商凭证（仅可删除 yaml 层，无法删除来自环境变量的提供商） */
export async function deleteProvider(name: string): Promise<void> {
  const res = await fetch(`${BASE}/providers/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `删除提供商失败: ${res.status}`);
  }
}
