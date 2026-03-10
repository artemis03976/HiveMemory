import { useState } from 'react';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { PanelRightOpen } from 'lucide-react';
import { GlobalNavBar } from './components/GlobalNavBar';
import { ContextSidebar } from './components/ContextSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { KernelVision } from './components/KernelVision';

function App() {
  const [isKernelVisionOpen, setIsKernelVisionOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* L1: Global Nav Bar */}
      <GlobalNavBar />

      {/* L2-L4: Resizable Layout */}
      <Group direction="horizontal" className="flex-1">
        {/* L2: Context Sidebar */}
        <Panel
          defaultSize={20}
          minSize={15}
          maxSize={30}
          className="min-w-[240px]"
        >
          <ContextSidebar />
        </Panel>

        <Separator className="w-px bg-white/10 hover:bg-primary/50 transition-colors" />

        {/* L3: Main Workspace */}
        <Panel defaultSize={isKernelVisionOpen ? 50 : 80} minSize={35}>
          <div className="relative h-full">
            <MainWorkspace />

            {/* Toggle Kernel Vision Button */}
            {!isKernelVisionOpen && (
              <button
                onClick={() => setIsKernelVisionOpen(true)}
                className="fixed top-4 right-4 p-3 rounded-lg glass-card hover:bg-purple-600/20 transition-all duration-200 cursor-pointer z-50"
                aria-label="Open Kernel Vision"
              >
                <PanelRightOpen className="w-5 h-5 text-foreground" />
              </button>
            )}
          </div>
        </Panel>

        {/* L4: Kernel Vision (conditional) */}
        {isKernelVisionOpen && (
          <>
            <Separator className="w-px bg-white/10 hover:bg-purple-500/50 transition-colors" />
            <Panel
              defaultSize={30}
              minSize={25}
              maxSize={40}
            >
              <KernelVision
                isOpen={isKernelVisionOpen}
                onClose={() => setIsKernelVisionOpen(false)}
              />
            </Panel>
          </>
        )}
      </Group>
    </div>
  );
}

export default App;
