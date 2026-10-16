/** 后端轮次重放为既有消息块；工具按 action_id 配对，不重建子帧的未回填输出。 */
import { normalizeMtpStatus } from './messageReducers';
import type { ContentBlock, Message, MtpAction } from '@/types';
import type { ConversationSession, ConversationTurn } from '@/services/sessionApi';

function blocksFor(turn: ConversationTurn): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  const events = [...turn.payload.turn_events].sort((a, b) => a.sequence - b.sequence);
  for (const event of events) {
    if (event.kind === 'assistant_message' && event.content) {
      blocks.push({ kind: 'text', text: event.content });
    } else if (event.kind === 'tool_call') {
      const result = events.find(e => e.kind === 'tool_result' && e.action_id === event.action_id);
      if (event.tool_kind === 'CALL') {
        blocks.push({
          kind: 'sub_agent', agentId: event.target || 'Agent',
          task: String(event.tool_args?.task || ''),
          status: result?.status === 'success' ? 'completed' : 'error',
          contentBlocks: [], finalText: result?.content || '调用未完成',
        });
      } else {
        const verb = event.tool_kind || 'UNKNOWN';
        const type = ['READ', 'SEARCH', 'RUN', 'WRITE', 'UPDATE'].includes(verb)
          ? verb as MtpAction['type'] : 'UNKNOWN';
        blocks.push({ kind: 'mtp', action: {
          id: event.action_id || `${turn.interaction_id}-${event.sequence}`, type,
          command: event.content || `${verb} ${event.target || ''}`,
          target: event.target || undefined, params: event.tool_args || undefined,
          status: result ? normalizeMtpStatus(result.status || 'error') : 'error',
          resultMessage: result?.content || '操作未完成',
        } });
      }
    }
  }
  if (!blocks.some(b => b.kind === 'text') && turn.payload.assistant_final_text) {
    blocks.push({ kind: 'text', text: turn.payload.assistant_final_text });
  }
  return blocks;
}
export function sessionMessages(session: ConversationSession): Message[] {
  return session.turns.flatMap(turn => [
    { id: `${turn.interaction_id}-user`, role: 'user' as const, content: turn.payload.user_message,
      timestamp: Date.parse(turn.created_at) },
    { id: `${turn.interaction_id}-assistant`, role: 'assistant' as const,
      content: turn.payload.assistant_final_text || '', contentBlocks: blocksFor(turn),
      timestamp: Date.parse(turn.created_at), agent_id: turn.agent_id,
      topicId: turn.topic_id, outcome: turn.payload.outcome, isStreaming: false },
  ]);
}
