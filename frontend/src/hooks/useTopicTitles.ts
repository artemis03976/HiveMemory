import { useEffect, useRef, useState } from 'react';
import { fetchTopics } from '@/services/topicApi';
import type { Message } from '@/types';

/**
 * 消息所在话题的标题，取自短期话题池；没有标题或已不在池中的话题不出现在结果里。
 * 只在出现尚未查过的话题时重新读取话题列表。
 */
export function useTopicTitles(messages: Message[]): Record<string, string> {
  const [titles, setTitles] = useState<Record<string, string>>({});
  const requested = useRef(new Set<string>());

  useEffect(() => {
    const missing = messages
      .map((m) => m.topicId)
      .filter((id): id is string => !!id && !requested.current.has(id));
    if (missing.length === 0) return;
    missing.forEach((id) => requested.current.add(id));
    fetchTopics()
      .then((topics) => {
        // 话题列表在没有标题时以 topic_id 充当标题，这里不把它当成标题
        const named = topics.filter((t) => t.title !== t.id);
        setTitles((prev) => ({ ...prev, ...Object.fromEntries(named.map((t) => [t.id, t.title])) }));
      })
      .catch(() => {
        // 读取失败时消息底部显示“未命名话题”；连接问题由页面顶部的状态条说明
      });
  }, [messages]);

  return titles;
}
