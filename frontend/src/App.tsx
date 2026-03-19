import { useState } from 'react';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { PanelRightOpen } from 'lucide-react';
import { GlobalNavBar } from './components/GlobalNavBar';
import type { ViewType } from './components/GlobalNavBar';
import { ContextSidebar } from './components/chat/ContextSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { SettingsPanel } from './components/SettingsPanel';
import { KernelVision } from './components/KernelVision';
import { MemoryGarden } from './components/MemoryGarden';

function App() {
  const [activeView, setActiveView] = useState<ViewType>('chat');
  const [isKernelVisionOpen, setIsKernelVisionOpen] = useState(false);

  // Render the main content based on active view
  const renderMainContent = () => {
    switch (activeView) {
      case 'chat':
        return <MainWorkspace />;
      case 'settings':
        return <SettingsPanel />;
      case 'kernel':
        return (
          <div className="h-screen flex items-center justify-center">
            <div className="text-foreground/60">Kernel Console - Coming Soon</div>
          </div>
        );
      case 'memory':
        return <MemoryGarden />;
      default:
        return <MainWorkspace />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* L1: Global Nav Bar */}
      <GlobalNavBar activeView={activeView} onViewChange={setActiveView} />

      {/* L2-L4: Resizable Layout */}
      {/* @ts-ignore - Group component type issue with direction prop */}
      <Group direction="horizontal" className="flex-1">
        {/* L2: Context Sidebar - Only show for chat view */}
        {activeView === 'chat' && (
          <>
            <Panel
              defaultSize={20}
              minSize={15}
              maxSize={30}
              className="min-w-[240px]"
            >
              <ContextSidebar />
            </Panel>
            <Separator className="w-px bg-white/10 hover:bg-primary/50 transition-colors" />
          </>
        )}

        {/* L3: Main Content Area */}
        <Panel defaultSize={80} minSize={35}>
          <div className="relative h-full flex">
            {/* Main content with dynamic width */}
            <div
              className="h-full transition-all duration-300 ease-in-out"
              style={{
                width: activeView === 'chat' && isKernelVisionOpen ? 'calc(100% - 400px)' : '100%'
              }}
            >
              {renderMainContent()}
            </div>

            {/* Toggle Kernel Vision Button - Only show for chat view */}
            {activeView === 'chat' && !isKernelVisionOpen && (
              <button
                onClick={() => setIsKernelVisionOpen(true)}
                className="fixed top-4 right-4 p-3 rounded-lg glass-card hover:bg-purple-600/20 transition-all duration-200 cursor-pointer z-50"
                aria-label="Open Kernel Vision"
              >
                <PanelRightOpen className="w-5 h-5 text-foreground" />
              </button>
            )}

            {/* L4: Kernel Vision - Slide-in panel */}
            {activeView === 'chat' && (
              <div
                className={`fixed right-0 top-0 h-full w-[400px] transition-transform duration-300 ease-in-out z-40 ${
                  isKernelVisionOpen ? 'translate-x-0' : 'translate-x-full'
                }`}
              >
                <KernelVision
                  onClose={() => setIsKernelVisionOpen(false)}
                />
              </div>
            )}
          </div>
        </Panel>
      </Group>
    </div>
  );
}

export default App;
