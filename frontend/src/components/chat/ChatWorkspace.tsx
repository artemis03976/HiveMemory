import TopBar from './TopBar';
import ChatMessage from './ChatMessage';
import OmniInput from './OmniInput';
import { useChatStore } from '@/stores';

interface ChatWorkspaceProps {
  activeTopicTitle: string;
}

export default function ChatWorkspace({ activeTopicTitle }: ChatWorkspaceProps) {
  const { messages } = useChatStore();

  return (
    <main className="flex-1 flex flex-col bg-surface-container-lowest relative h-full overflow-hidden">
      <TopBar activeTopicTitle={activeTopicTitle} />

      <div className="flex-1 overflow-y-auto p-8 space-y-8 max-w-4xl mx-auto w-full scrollbar-hide">
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            少女祈祷中...
          </div>
        )}
      </div>

      <OmniInput />
    </main>
  );
}
