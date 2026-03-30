/**
 * Topic API Client
 *
 * Handles HTTP requests to the backend topic management API.
 */

import type { Topic } from '@/types';

const DEFAULT_USER_ID = 'default';

interface TopicListResponse {
  topics: Array<{
    topic_id: string;
    title: string;
    state_summary: string;
    last_turn: { role: string; content: string } | null;
    total_tokens: number;
  }>;
}

function mapTopic(raw: TopicListResponse['topics'][number]): Topic {
  return {
    id: raw.topic_id,
    title: raw.title,
    summary: raw.state_summary,
    activeNow: true, // We map status to activeNow since Topic has activeNow, timeAgo, model etc. in new-frontend
    model: 'GPT-4o', // Default model for now
    lastActive: Date.now(),
    messageCount: 0,
    totalTokens: raw.total_tokens,
  };
}

export async function fetchTopics(userId: string = DEFAULT_USER_ID): Promise<Topic[]> {
  const res = await fetch(`/api/v1/topics?user_id=${encodeURIComponent(userId)}`);
  if (!res.ok) throw new Error(`fetchTopics failed: ${res.status}`);
  const data: TopicListResponse = await res.json();
  return data.topics.map(mapTopic);
}

export async function archiveTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}/trigger`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`archiveTopic failed: ${res.status}`);
}

export async function deleteTopic(topicId: string): Promise<void> {
  const res = await fetch(`/api/v1/topics/${encodeURIComponent(topicId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`deleteTopic failed: ${res.status}`);
}
