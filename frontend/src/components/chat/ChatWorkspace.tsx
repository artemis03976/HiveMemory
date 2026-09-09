import { useRef, useState } from 'react';
import { FileUp } from 'lucide-react';
import TopBar from './TopBar';
import ChatMessage from './ChatMessage';
import OmniInput from './OmniInput';
import { useChatStore } from '@/stores';
import { useAttachmentUpload } from '@/hooks/useAttachmentUpload';

interface ChatWorkspaceProps {
  activeTopicTitle: string;
}

export default function ChatWorkspace({ activeTopicTitle }: ChatWorkspaceProps) {
  const { messages } = useChatStore();
  const { enqueueFiles } = useAttachmentUpload();
  const [isDragActive, setIsDragActive] = useState(false);
  // 拖拽进入子元素会连续触发 dragenter/dragleave，用计数判断是否真正离开工作区
  const dragDepthRef = useRef(0);

  const hasFiles = (e: React.DragEvent) =>
    Array.from(e.dataTransfer?.types ?? []).includes('Files');

  const handleDragEnter = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setIsDragActive(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    // 必须 preventDefault，否则浏览器会直接打开拖入的文件
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragActive(false);
    // 只接收 DataTransfer.files：目录和无文件 payload 直接忽略
    enqueueFiles(e.dataTransfer.files);
  };

  return (
    <main
      className="flex-1 flex flex-col bg-surface-container-lowest relative h-full overflow-hidden"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
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

      {/* 拖入文件时的 drop 状态提示；pointer-events-none 保证 drop 命中工作区 */}
      {isDragActive && (
        <div className="absolute inset-0 z-40 pointer-events-none flex items-center justify-center bg-primary/10 backdrop-blur-sm border-2 border-dashed border-primary/50 rounded-lg">
          <div className="flex flex-col items-center gap-2 text-primary">
            <FileUp className="w-10 h-10" />
            <span className="text-sm font-medium">松开以上传附件</span>
          </div>
        </div>
      )}
    </main>
  );
}
