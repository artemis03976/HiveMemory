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
import type {
  ChatDoneEvent,
  GenerationIdEvent,
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
  TopicInfoEvent,
} from '@/types';

interface CreateChatSSECallbacksDeps {
  assistantMessageId: string;
  updateMessages: (updater: (messages: Message[]) => Message[]) => void;
  setTopicInfo: (data: TopicInfoEvent) => void;
  setRetrievedMemories: (memories: MemoryAtom[]) => void;
  handleCommandResult: (data: CommandResultEvent) => void;
  setGenerationId: (data: GenerationIdEvent) => void;
  setRunStatus: (data: ChatRunStatusEvent) => void;
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

    onTopicInfo: (data: TopicInfoEvent) => {
      deps.setTopicInfo(data);
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
      const errorMessage = data.message || '系统错误，请检查后端服务器';
      deps.finalizeError(errorMessage);
    },

    onConnectionError: (error) => {
      const errorMessage = '系统错误，请检查后端服务器';
      deps.finalizeError(errorMessage, error.message);
    },

    onGenerationId: (data: GenerationIdEvent) => {
      deps.setGenerationId(data);
    },

    onRunStatus: (data: ChatRunStatusEvent) => {
      deps.setRunStatus(data);
    },
  };
}
