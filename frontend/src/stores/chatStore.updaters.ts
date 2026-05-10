import type { ContentBlock, InlineBlock, Message, MtpAction, SubAgentBlock } from '@/types';

function updateAssistantMessage(
  messages: Message[],
  assistantMessageId: string,
  updater: (msg: Message) => Message,
): Message[] {
  return messages.map((msg) => (msg.id === assistantMessageId ? updater(msg) : msg));
}

type MtpResultPayload = {
  status: string;
  verb?: string;
  target?: string;
  args?: Record<string, unknown>;
  raw_text?: string;
  result_message?: string;
  stats?: Record<string, unknown>;
};

/** 向顶层消息块末尾追加 token，仅用于 assistant 主消息流。 */
export function appendContentToken(blocks: ContentBlock[], token: string): ContentBlock[] {
  const updated = [...blocks];
  const last = updated[updated.length - 1];
  if (last && last.kind === 'text') {
    updated[updated.length - 1] = { kind: 'text', text: last.text + token };
  } else {
    updated.push({ kind: 'text', text: token });
  }
  return updated;
}

/** 向子代理线性内容流末尾追加 token，仅允许 text/mtp。 */
export function appendInlineToken(blocks: InlineBlock[], token: string): InlineBlock[] {
  const updated = [...blocks];
  const last = updated[updated.length - 1];
  if (last && last.kind === 'text') {
    updated[updated.length - 1] = { kind: 'text', text: last.text + token };
  } else {
    updated.push({ kind: 'text', text: token });
  }
  return updated;
}

/** Normalize stream delta to avoid duplicated prefix/overlap chunks */
export function normalizeStreamDelta(currentText: string, incoming: string): string {
  if (!incoming) return '';
  if (!currentText) return incoming;
  if (currentText.endsWith(incoming)) return '';
  if (incoming.startsWith(currentText)) return incoming.slice(currentText.length);

  const maxOverlap = Math.min(currentText.length, incoming.length);
  for (let k = maxOverlap; k > 0; k--) {
    if (currentText.slice(-k) === incoming.slice(0, k)) {
      return incoming.slice(k);
    }
  }
  return incoming;
}

/** 在顶层消息块中追加 MTP 卡片。 */
export function pushContentMtpBlock(blocks: ContentBlock[], action: MtpAction): ContentBlock[] {
  return [...blocks, { kind: 'mtp', action }];
}

/** 在子代理线性内容流中追加 MTP 卡片。 */
export function pushInlineMtpBlock(blocks: InlineBlock[], action: MtpAction): InlineBlock[] {
  return [...blocks, { kind: 'mtp', action }];
}

/** 更新顶层消息块中最后一个 MTP 卡片状态。 */
export function updateLastContentMtpStatus(
  blocks: ContentBlock[],
  payload: MtpResultPayload,
): ContentBlock[] {
  const updated = [...blocks];
  for (let i = updated.length - 1; i >= 0; i--) {
    const b = updated[i];
    if (b.kind === 'mtp') {
      const verb = normalizeVerb(payload.verb || b.action.type || 'UNKNOWN');
      const command =
        payload.raw_text || [verb, payload.target].filter(Boolean).join(' | ') || b.action.command;
      updated[i] = {
        kind: 'mtp',
        action: {
          ...b.action,
          type: verb,
          command,
          target: payload.target ?? b.action.target,
          params: payload.args ?? b.action.params,
          status: normalizeMtpStatus(payload.status),
          resultMessage: payload.result_message ?? b.action.resultMessage,
          stats: payload.stats ?? b.action.stats,
        },
      };
      break;
    }
  }
  return updated;
}

/** 更新子代理线性内容流中最后一个 MTP 卡片状态。 */
export function updateLastInlineMtpStatus(
  blocks: InlineBlock[],
  payload: MtpResultPayload,
): InlineBlock[] {
  const updated = [...blocks];
  for (let i = updated.length - 1; i >= 0; i--) {
    const b = updated[i];
    if (b.kind === 'mtp') {
      const verb = normalizeVerb(payload.verb || b.action.type || 'UNKNOWN');
      const command =
        payload.raw_text || [verb, payload.target].filter(Boolean).join(' | ') || b.action.command;
      updated[i] = {
        kind: 'mtp',
        action: {
          ...b.action,
          type: verb,
          command,
          target: payload.target ?? b.action.target,
          params: payload.args ?? b.action.params,
          status: normalizeMtpStatus(payload.status),
          resultMessage: payload.result_message ?? b.action.resultMessage,
          stats: payload.stats ?? b.action.stats,
        },
      };
      break;
    }
  }
  return updated;
}

export function normalizeVerb(verb?: string): MtpAction['type'] {
  const upper = (verb || '').toUpperCase();
  if (['RUN', 'READ', 'SEARCH', 'WRITE', 'UPDATE'].includes(upper)) {
    return upper as MtpAction['type'];
  }
  return 'UNKNOWN';
}

export function normalizeMtpStatus(status?: string): MtpAction['status'] {
  const lower = (status || '').toLowerCase();
  if (lower === 'pending' || lower === 'executing' || lower === 'success' || lower === 'error') {
    return lower;
  }
  return 'executing';
}

/** Push a new SubAgentBlock, cutting the current text segment */
export function pushSubAgentBlock(blocks: ContentBlock[], agentId: string, task: string): ContentBlock[] {
  return [...blocks, { kind: 'sub_agent', agentId, task, status: 'running', contentBlocks: [] }];
}

/** Update the last SubAgentBlock in contentBlocks via an updater function */
export function updateLastSubAgentBlock(
  blocks: ContentBlock[],
  updater: (sub: SubAgentBlock) => SubAgentBlock,
): ContentBlock[] {
  const updated = [...blocks];
  for (let i = updated.length - 1; i >= 0; i--) {
    if (updated[i].kind === 'sub_agent') {
      updated[i] = updater(updated[i] as SubAgentBlock);
      break;
    }
  }
  return updated;
}

/** Rebuild contentBlocks from final_text while preserving MTP blocks in-place */
export function rebuildBlocksWithFinalText(blocks: ContentBlock[], finalText: string): ContentBlock[] {
  const hasNonText = blocks.some((b) => b.kind === 'mtp' || b.kind === 'sub_agent');
  if (!hasNonText) {
    return [{ kind: 'text', text: finalText }];
  }

  const textSlots: { index: number; streamLen: number }[] = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    if (block.kind === 'text') {
      textSlots.push({ index: i, streamLen: block.text.length });
    }
  }

  const totalStreamLen = textSlots.reduce((s, t) => s + t.streamLen, 0);

  if (totalStreamLen === 0) {
    const result: ContentBlock[] = blocks.map((b) => (b.kind === 'text' ? { kind: 'text', text: '' } : b));
    result.push({ kind: 'text', text: finalText });
    return result;
  }

  const result: ContentBlock[] = [...blocks];
  let cursor = 0;
  for (let si = 0; si < textSlots.length; si++) {
    const slot = textSlots[si];
    const isLast = si === textSlots.length - 1;
    const charCount = isLast
      ? finalText.length - cursor
      : Math.round((slot.streamLen / totalStreamLen) * finalText.length);
    result[slot.index] = { kind: 'text', text: finalText.slice(cursor, cursor + charCount) };
    cursor += charCount;
  }

  return result;
}

export function applyAssistantToken(messages: Message[], assistantMessageId: string, incoming: string): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => {
    const delta = normalizeStreamDelta(msg.content, incoming);
    if (!delta) return msg;
    return {
      ...msg,
      content: msg.content + delta,
      contentBlocks: appendContentToken(msg.contentBlocks || [], delta),
    };
  });
}

export function applyAssistantMtpStart(
  messages: Message[],
  assistantMessageId: string,
  payload: { verb?: string; target?: string; args?: Record<string, unknown>; raw_text?: string },
): Message[] {
  const verb = normalizeVerb(payload.verb);
  const newAction: MtpAction = {
    id: crypto.randomUUID(),
    type: verb,
    command: payload.raw_text || [verb, payload.target].filter(Boolean).join(' | ') || verb,
    target: payload.target,
    params: payload.args,
    status: 'executing',
    timestamp: Date.now(),
  };
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: pushContentMtpBlock(msg.contentBlocks || [], newAction),
  }));
}

export function applyAssistantMtpResult(
  messages: Message[],
  assistantMessageId: string,
  payload: {
    status: string;
    verb?: string;
    target?: string;
    args?: Record<string, unknown>;
    raw_text?: string;
    result_message?: string;
    stats?: Record<string, unknown>;
  },
): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: updateLastContentMtpStatus(msg.contentBlocks || [], {
      ...payload,
      verb: normalizeVerb(payload.verb),
    }),
  }));
}

export function applySubAgentStart(
  messages: Message[],
  assistantMessageId: string,
  payload: { agent_id: string; task: string },
): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: pushSubAgentBlock(msg.contentBlocks || [], payload.agent_id, payload.task),
  }));
}

export function applySubAgentToken(messages: Message[], assistantMessageId: string, incoming: string): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: updateLastSubAgentBlock(msg.contentBlocks || [], (sub) => ({
      ...sub,
      contentBlocks: appendInlineToken(sub.contentBlocks, incoming),
    })),
  }));
}

export function applySubAgentMtpStart(
  messages: Message[],
  assistantMessageId: string,
  payload: { verb?: string; target?: string; args?: Record<string, unknown>; raw_text?: string },
): Message[] {
  const verb = normalizeVerb(payload.verb);
  const newAction: MtpAction = {
    id: crypto.randomUUID(),
    type: verb,
    command: payload.raw_text || [verb, payload.target].filter(Boolean).join(' | ') || verb,
    target: payload.target,
    params: payload.args,
    status: 'executing',
    timestamp: Date.now(),
  };

  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: updateLastSubAgentBlock(msg.contentBlocks || [], (sub) => ({
      ...sub,
      contentBlocks: pushInlineMtpBlock(sub.contentBlocks, newAction),
    })),
  }));
}

export function applySubAgentMtpResult(
  messages: Message[],
  assistantMessageId: string,
  payload: {
    status: string;
    verb?: string;
    target?: string;
    args?: Record<string, unknown>;
    raw_text?: string;
    result_message?: string;
    stats?: Record<string, unknown>;
  },
): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: updateLastSubAgentBlock(msg.contentBlocks || [], (sub) => ({
      ...sub,
      contentBlocks: updateLastInlineMtpStatus(sub.contentBlocks, {
        ...payload,
        verb: normalizeVerb(payload.verb),
      }),
    })),
  }));
}

export function applySubAgentEnd(
  messages: Message[],
  assistantMessageId: string,
  payload: { status: 'success' | 'error'; final_text?: string },
): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    contentBlocks: updateLastSubAgentBlock(msg.contentBlocks || [], (sub) => ({
      ...sub,
      status: payload.status === 'success' ? 'completed' : 'error',
      finalText: payload.final_text,
    })),
  }));
}

export function applyDone(messages: Message[], assistantMessageId: string, finalText: string): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    content: finalText,
    contentBlocks: (msg.contentBlocks || []).some((b) => b.kind === 'mtp' || b.kind === 'sub_agent')
      ? msg.contentBlocks
      : rebuildBlocksWithFinalText(msg.contentBlocks || [], finalText),
    isStreaming: false,
  }));
}

export function applyStreamError(
  messages: Message[],
  assistantMessageId: string,
  errorMessage: string,
  errorDetail?: string,
): Message[] {
  return updateAssistantMessage(messages, assistantMessageId, (msg) => ({
    ...msg,
    content: errorMessage,
    contentBlocks: [{ kind: 'text', text: errorMessage }],
    isStreaming: false,
    error: errorDetail ?? errorMessage,
  }));
}
