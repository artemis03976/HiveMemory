import { motion, AnimatePresence } from 'motion/react';
import { AlertTriangle } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useDialog } from '@/hooks/useDialog';

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
  isDestructive?: boolean;
}

export function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消',
  onConfirm,
  onCancel,
  isDestructive = true,
}: ConfirmDialogProps) {
  const { panelRef, onKeyDown } = useDialog<HTMLDivElement>(isOpen, onCancel);
  if (typeof document === 'undefined') return null;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-60 flex items-center justify-center p-4 bg-scrim backdrop-blur-sm"
          onClick={onCancel}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            ref={panelRef}
            role="alertdialog"
            aria-modal="true"
            aria-label={title}
            tabIndex={-1}
            onKeyDown={onKeyDown}
            className="relative w-full max-w-md bg-surface-container-lowest border border-line rounded-2xl shadow-pop overflow-hidden outline-none"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-full shrink-0 ${isDestructive ? 'bg-magic-fire/10 text-magic-fire' : 'bg-primary/10 text-primary'}`}>
                  <AlertTriangle className="w-6 h-6" aria-hidden="true" />
                </div>
                <div className="flex-1 min-w-0 mt-1">
                  <h3 className="text-lg font-bold text-fg mb-2">
                    {title}
                  </h3>
                  <p className="text-sm text-fg-3 leading-relaxed">
                    {message}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-end gap-3 p-4">
              <button
                onClick={onCancel}
                className="px-4 py-2 rounded-lg text-sm font-medium text-fg-2 hover:text-fg hover:bg-ink/5 transition-colors"
              >
                {cancelText}
              </button>
              <button
                onClick={() => {
                  onConfirm();
                  onCancel();
                }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isDestructive 
                    ? 'bg-magic-fire/20 text-magic-fire hover:bg-magic-fire/30 border border-magic-fire/20'
                    : 'bg-primary/20 text-primary hover:bg-primary/30 border border-primary/20'
                }`}
              >
                {confirmText}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}
