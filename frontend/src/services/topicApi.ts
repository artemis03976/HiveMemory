/**
 * Topic API Client
 *
 * Handles HTTP requests to the backend topic management API.
 *
 * 身份约定：Topic 请求统一携带 `user_id + workspace_id`
 * 基础选择（x-user-id / x-workspace-id 请求头）；旧 `?user_id=` query
 * 兼容入口已不再由前端使用。
 */
import { apiErrorMessage } from '@/services/apiError';

import { identityHeaders } from '@/services/identity';
import type { Topic } from '@/types';

export interface ApiTopicSnapshot {
  topic_id: string;
  topic_title?: string;
  title?: string;
  topic_summary?: string;
  state_summary?: string;
  last_turn?: { user: string; assistant: string } | null;
  block_count?: number;
  /** 话题最近一次写入的 Unix 时间戳（秒）。 */
  last_update: number;
  total_tokens?: number;
  model_used?: string;
}

interface ActiveTopicListResponse {
  topics: ApiTopicSnapshot[];
}

export function mapTopic(raw: ApiTopicSnapshot): Topic {
  const title = raw.topic_title || raw.title || raw.topic_id;

  return {
    id: raw.topic_id,
    title,
    summary: raw.state_summary || raw.topic_summary || undefined,
    // 从后端 model_used 字段读取真实使用的模型名，未设置时回落到默认文本
    model: raw.model_used || 'Unknown',
    lastActive: raw.last_update * 1000,
    messageCount: raw.block_count ?? 0,
    totalTokens: raw.total_tokens ?? 0,
  };
}

export async function fetchTopics(): Promise<Topic[]> {
  const res = await fetch('/api/v1/topics', {
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchTopics failed'));
  const data: ActiveTopicListResponse = await res.json();
  return data.topics.map(mapTopic);
}

export async function settleTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}/settle`, {
    method: 'POST',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'settleTopic failed'));
}

export async function deleteTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}`, {
    method: 'DELETE',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'deleteTopic failed'));
}
