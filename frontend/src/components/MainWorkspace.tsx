import { useEffect, useRef, useState } from 'react';
import { ChatMessage } from './chat/ChatMessage';
import { OmniInput } from './chat/OmniInput';
import type { Message } from '@/types';

const mockMessages: Message[] = [
  {
    id: 'm1',
    role: 'assistant',
    content: '你好！有什么可以帮您？',
    timestamp: Date.now() - 60000,
  },
  {
    id: 'm2',
    role: 'user',
    content: '帮我创建一个新的前端项目',
    timestamp: Date.now() - 30000,
  },
  {
    id: 'm3',
    role: 'assistant',
    content: '好的，我来帮你创建前端项目。首先让我检查一下当前的项目结构...',
    timestamp: Date.now() - 20000,
    mtpActions: [
      {
        id: 'a1',
        type: 'RUN',
        command: 'sys_read_file',
        params: { path: 'package.json' },
        status: 'success',
        response: '{\n  "name": "frontend",\n  "version": "0.0.0",\n  "type": "module"\n}',
        timestamp: Date.now() - 19000,
      },
    ],
  },
  {
    id: 'm_code_test',
    role: 'assistant',
    content: `这里是一段 Python 代码示例：

\`\`\`python
def hello_world():
    print("Hello, HiveMemory!")
    return True
\`\`\`

`,
    timestamp: Date.now() - 10000,
  },
];

export function MainWorkspace() {
  const [messages, setMessages] = useState<Message[]>(mockMessages);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (content: string) => {
    const newMessage: Message = {
      id: `m${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    };
    setMessages([...messages, newMessage]);

    // Simulate assistant response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: `m${Date.now()}`,
        role: 'assistant',
        content: '收到！让我处理一下...',
        timestamp: Date.now(),
        mtpActions: [
          {
            id: `a${Date.now()}`,
            type: 'RUN',
            command: 'sys_execute',
            status: 'executing',
            timestamp: Date.now(),
          },
        ],
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }, 1000);
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Chat Stream */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
        <div className="max-w-4xl mx-auto">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Omni Input - Suspended Floating Design */}
      <div className="p-6 relative z-10">
        <div className="max-w-4xl mx-auto">
          <OmniInput onSend={handleSendMessage} />
        </div>
      </div>
    </div>
  );
}
