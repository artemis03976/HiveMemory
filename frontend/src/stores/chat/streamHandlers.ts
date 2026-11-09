import {
  applyAssistantMtpResult,
  applyAssistantMtpStart,
  applyAssistantToken,
  applySubAgentEnd,
  applySubAgentMtpResult,
  applySubAgentMtpStart,
  applySubAgentStart,
  applySubAgentToken,
} from '@/stores/chat/messageReducers';
import { ApiError } from '@/services/apiError';
import { chatErrorMessage } from '@/stores/chat/failureMessage';
import type {
  ChatDoneEvent,
  ContextCompactionEvent,
  ProcessIdEvent,
  ChatRunStatusEvent,
  CommandResultEvent,
  MemoryAtom,
  MemoryRefsEvent,
  Message,
  MTPResultEvent,
  MTPStartEvent,
  SSECallbacks,
  SubAgentEndEvent,
  SubAgentStartEvent,
} from '@/types';

interface CreateChatSSECallbacksDeps {
  assistantMessageId: string;
  updateMessages: (updater: (messages: Message[]) => Message[]) => void;
  setRetrievedMemories: (memories: MemoryAtom[]) => void;
  handleCommandResult: (data: CommandResultEvent) => void;
  setProcessId: (data: ProcessIdEvent) => void;
  setRunStatus: (data: ChatRunStatusEvent) => void;
  setCompactionProgress: (data: ContextCompactionEvent) => void;
  markStreaming: () => void;
  finalizeSuccess: (data: ChatDoneEvent) => void;
  finalizeError: (errorMessage: string, errorDetail?: string) => void;
}

export function createChatSSECallbacks(deps: CreateChatSSECallbacksDeps): SSECallbacks {
  const { assistantMessageId } = deps;

  return {
    onToken: (data) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applyAssistantToken(messages, assistantMessageId, data.content));
    },

    onMTPStart: (data: MTPStartEvent) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applyAssistantMtpStart(messages, assistantMessageId, data));
    },

    onMTPResult: (data: MTPResultEvent) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applyAssistantMtpResult(messages, assistantMessageId, data));
    },

    onSubAgentStart: (data: SubAgentStartEvent) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applySubAgentStart(messages, assistantMessageId, data));
    },

    onSubAgentToken: (data) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applySubAgentToken(messages, assistantMessageId, data.content));
    },

    onSubAgentMTPStart: (data: MTPStartEvent) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applySubAgentMtpStart(messages, assistantMessageId, data));
    },

    onSubAgentMTPResult: (data: MTPResultEvent) => {
      deps.markStreaming();
      deps.updateMessages((messages) => applySubAgentMtpResult(messages, assistantMessageId, data));
    },

    onSubAgentEnd: (data: SubAgentEndEvent) => {
      deps.updateMessages((messages) => applySubAgentEnd(messages, assistantMessageId, data));
    },

    onMemoryRefs: (data: MemoryRefsEvent) => {
      const memories = Array.isArray(data.memories) ? data.memories : [];
      deps.setRetrievedMemories(memories);
    },

    onCommandResult: (data: CommandResultEvent) => {
      deps.handleCommandResult(data);
    },

    onDone: (data: ChatDoneEvent) => {
      deps.finalizeSuccess(data);
    },

    onError: (data) => {
      deps.finalizeError(chatErrorMessage(data.code, data.message));
    },

    onConnectionError: (error) => {
      // 后端在开流之前拒绝（准入、会话占用、管理凭据等）时，它给出的文案就是原因；
      // 只有连不上后端或流中断才是系统错误。
      if (error instanceof ApiError) {
        deps.finalizeError(error.message);
        return;
      }
      deps.finalizeError('系统错误，请检查后端服务器', error.message);
    },

    onProcessId: (data: ProcessIdEvent) => {
      deps.setProcessId(data);
    },

    onRunStatus: (data: ChatRunStatusEvent) => {
      deps.setRunStatus(data);
    },

    onContextCompaction: (data) => {
      deps.setCompactionProgress(data);
    },
  };
}
