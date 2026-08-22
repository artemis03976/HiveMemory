/**
 * Topic API Client
 *
 * Handles HTTP requests to the backend topic management API.
 */

import { DEFAULT_USER_ID } from '@/constants/identity';
import type { Topic } from '@/types';

export interface ApiTopicSnapshot {
  topic_id: string;
  topic_title?: string;
  title?: string;
  topic_summary?: string;
  state_summary?: string;
  last_turn?: { role: string; content: string } | Record<string, string> | null;
  block_count?: number;
  last_accessed_at?: number;
  total_tokens?: number;
  model_used?: string;
}

interface TopicListResponse {
  topics: ApiTopicSnapshot[];
}

export function mapTopic(raw: ApiTopicSnapshot, activeTopicId?: string): Topic {
  const title = raw.topic_title || raw.title || raw.topic_id;

  return {
    id: raw.topic_id,
    title,
    summary: raw.state_summary || raw.topic_summary || undefined,
    activeNow: activeTopicId ? raw.topic_id === activeTopicId : true,
    // 从后端 model_used 字段读取真实使用的模型名，未设置时回落到默认文本
    model: raw.model_used || 'Unknown',
    lastActive: raw.last_accessed_at ? raw.last_accessed_at * 1000 : Date.now(),
    messageCount: raw.block_count ?? 0,
    totalTokens: raw.total_tokens ?? 0,
  };
}

export async function fetchTopics(userId: string = DEFAULT_USER_ID): Promise<Topic[]> {
  const res = await fetch(`/api/v1/topics?user_id=${encodeURIComponent(userId)}`);
  if (!res.ok) throw new Error(`fetchTopics failed: ${res.status}`);
  const data: TopicListResponse = await res.json();
  return data.topics.map((topic) => mapTopic(topic));
}

export async function settleTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}/settle`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`settleTopic failed: ${res.status}`);
}

export async function deleteTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`deleteTopic failed: ${res.status}`);
}
