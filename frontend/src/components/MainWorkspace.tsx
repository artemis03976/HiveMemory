import { useEffect, useRef } from 'react';
import { ChatMessage } from './chat/ChatMessage';
import { OmniInput } from './chat/OmniInput';
import { useChatStore } from '@/stores/chatStore';

export function MainWorkspace() {
  const messages = useChatStore((state) => state.messages);
  const isStreaming = useChatStore((state) => state.isStreaming);
  const sendMessage = useChatStore((state) => state.sendMessage);
  const clearMessages = useChatStore((state) => state.clearMessages);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    await sendMessage(content);
  };

  const handleClearMessages = () => {
    clearMessages();
  };

  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
        <div className="max-w-4xl mx-auto">
          <div className="mb-4 flex justify-end">
            <button
              type="button"
              onClick={handleClearMessages}
              disabled={isStreaming || messages.length === 0}
              className="px-3 py-1.5 text-xs rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              清空对话
            </button>
          </div>
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              isStreaming={message.isStreaming}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="p-6 relative z-10">
        <div className="max-w-4xl mx-auto">
          <OmniInput onSend={handleSendMessage} disabled={isStreaming} />
        </div>
      </div>
    </div>
  );
}
