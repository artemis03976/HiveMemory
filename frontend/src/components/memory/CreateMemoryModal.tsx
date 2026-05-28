import { useState } from 'react';
import { X, Plus, AtSign } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useToastStore } from '@/stores/toastStore';
import { createMemory } from '@/services/memoryApi';
import { useDraft } from '@/hooks/useDraft';
import { memoryTypeLabels, type MemoryType } from '@/types/memory';

interface CreateMemoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreated: () => void;
}

const CREATABLE_TYPES: MemoryType[] = [
  'CODE_SNIPPET', 'FACT', 'URL_RESOURCE',
  'REFLECTION', 'USER_PROFILE', 'WORK_IN_PROGRESS',
];

interface FormState {
  title: string;
  summary: string;
  content: string;
  memory_type: MemoryType | '';
  tags: string;
  alias: string;
}

const INITIAL_FORM: FormState = {
  title: '',
  summary: '',
  content: '',
  memory_type: '',
  tags: '',
  alias: '',
};

interface FormErrors {
  title?: string;
  summary?: string;
  content?: string;
  memory_type?: string;
}

function validate(form: FormState): FormErrors {
  const errors: FormErrors = {};
  const summaryLength = form.summary.trim().length;

  if (!form.title.trim()) errors.title = '标题不能为空';
  if (summaryLength < 10) errors.summary = '摘要至少需要 10 个字符';
  if (summaryLength > 500) errors.summary = '摘要不能超过 500 个字符';
  if (!form.content.trim()) errors.content = '内容不能为空';
  if (!form.memory_type) errors.memory_type = '请选择记忆类型';

  return errors;
}

export default function CreateMemoryModal({ isOpen, onClose, onCreated }: CreateMemoryModalProps) {
  const [errors, setErrors] = useState<FormErrors>({});
  const addToast = useToastStore(s => s.addToast);

  const {
    draft: form,
    isSaving,
    updateDraft,
    submit,
    reset,
  } = useDraft({
    initialData: INITIAL_FORM,
    skipUnchangedSubmit: false,
    onSave: async (draftData) => {
      const tags = draftData.tags
        .split(',')
        .map(t => t.trim())
        .filter(Boolean);

      await createMemory({
        title: draftData.title.trim(),
        summary: draftData.summary.trim(),
        content: draftData.content,
        memory_type: draftData.memory_type,
        tags,
        alias: draftData.alias.trim() || null,
      });
    },
    onSuccess: () => {
      addToast('记忆创建成功', 'success');
      reset();
      onCreated();
    },
    onError: (err) => {
      const msg = err instanceof Error ? err.message : '创建失败';
      addToast(`创建失败: ${msg}`, 'error');
    },
  });

  const update = (patch: Partial<FormState>) => {
    updateDraft(patch);
    setErrors(prev => {
      const next = { ...prev };
      for (const key of Object.keys(patch) as (keyof FormErrors)[]) {
        delete next[key];
      }
      return next;
    });
  };

  const handleClose = () => {
    reset();
    setErrors({});
    onClose();
  };

  const handleSubmit = async () => {
    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    await submit({ force: true });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 flex items-center justify-center p-4 sm:p-6 md:p-12 bg-black/60 backdrop-blur-sm"
          onMouseDown={(e) => { if (e.target === e.currentTarget) handleClose(); }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="relative w-full max-w-2xl max-h-[90vh] flex flex-col bg-surface-container-lowest border border-white/10 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/5 bg-surface-dim shrink-0">
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Plus className="w-5 h-5 text-primary" />
                注入新记忆
              </h2>
              <button onClick={handleClose} className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Form Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-5 scrollbar-hide">
              {/* Title */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">标题 *</label>
                <input
                  className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  placeholder="记忆标题..."
                  value={form.title}
                  onChange={e => update({ title: e.target.value })}
                />
                {errors.title && <span className="text-xs text-red-400 ml-1">{errors.title}</span>}
              </div>

              {/* Memory Type */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">记忆类型 *</label>
                <select
                  className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  value={form.memory_type}
                  onChange={e => update({ memory_type: e.target.value as MemoryType | '' })}
                >
                  <option value="" disabled>选择类型...</option>
                  {CREATABLE_TYPES.map(t => (
                    <option key={t} value={t}>{memoryTypeLabels[t]}</option>
                  ))}
                </select>
                {errors.memory_type && <span className="text-xs text-red-400 ml-1">{errors.memory_type}</span>}
              </div>

              {/* Summary */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">摘要 * <span className="text-slate-600 normal-case">(10-500 字符)</span></label>
                <textarea
                  className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none"
                  rows={2}
                  placeholder="简要描述这条记忆的核心内容..."
                  value={form.summary}
                  onChange={e => update({ summary: e.target.value })}
                />
                {errors.summary && <span className="text-xs text-red-400 ml-1">{errors.summary}</span>}
              </div>

              {/* Content */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">内容 * <span className="text-slate-600 normal-case">(支持 Markdown)</span></label>
                <textarea
                  className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none font-mono text-sm"
                  rows={8}
                  placeholder="记忆的详细内容..."
                  value={form.content}
                  onChange={e => update({ content: e.target.value })}
                />
                {errors.content && <span className="text-xs text-red-400 ml-1">{errors.content}</span>}
              </div>

              {/* Tags */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">标签 <span className="text-slate-600 normal-case">(逗号分隔)</span></label>
                <input
                  className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  placeholder="tag1, tag2, tag3"
                  value={form.tags}
                  onChange={e => update({ tags: e.target.value })}
                />
              </div>

              {/* Alias */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[11px] uppercase tracking-wider text-slate-500 font-medium ml-1">别名 <span className="text-slate-600 normal-case">(可选)</span></label>
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-black/20 border border-white/10 focus-within:border-primary/50 focus-within:ring-1 focus:ring-primary/50 transition-all">
                  <AtSign className="w-4 h-4 text-slate-500 shrink-0" />
                  <input
                    className="flex-1 text-sm font-mono bg-transparent text-primary/90 placeholder:text-slate-600 focus:outline-none"
                    placeholder="e.g. my-memory-alias"
                    value={form.alias}
                    onChange={e => update({ alias: e.target.value })}
                  />
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 p-6 border-t border-white/5 bg-surface-dim shrink-0">
              <button
                onClick={handleClose}
                className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 rounded-lg hover:bg-white/5 transition-colors"
              >
                取消
              </button>
              <button
                onClick={handleSubmit}
                disabled={isSaving}
                className="flex items-center gap-2 px-5 py-2 text-sm font-bold text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSaving ? '创建中...' : '创建记忆'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
