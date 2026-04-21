import { useState, useCallback, useMemo, useEffect } from 'react';

// 简单的深度比较函数，适用于表单数据
function deepEqual(obj1: any, obj2: any): boolean {
  if (obj1 === obj2) return true;

  if (typeof obj1 !== 'object' || typeof obj2 !== 'object' || obj1 == null || obj2 == null) {
    return false;
  }

  const keys1 = Object.keys(obj1);
  const keys2 = Object.keys(obj2);

  if (keys1.length !== keys2.length) return false;

  for (const key of keys1) {
    if (!keys2.includes(key)) return false;
    if (!deepEqual(obj1[key], obj2[key])) return false;
  }

  return true;
}

interface UseDraftOptions<T> {
  initialData: T;
  onSave: (draftData: T) => Promise<void>;
  onSuccess?: () => void;
  onError?: (error: unknown) => void;
}

export function useDraft<T>({ initialData, onSave, onSuccess, onError }: UseDraftOptions<T>) {
  const [draft, setDraft] = useState<T>(initialData);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // 当外部传入的初始数据变化时，同步更新草稿（例如切换了选中的 Agent）
  useEffect(() => {
    setDraft(initialData);
  }, [initialData]);

  // 自动计算是否被修改过
  const isDirty = useMemo(() => !deepEqual(initialData, draft), [initialData, draft]);

  // 局部更新草稿的函数
  const updateDraft = useCallback((updates: Partial<T>) => {
    setDraft(prev => ({ ...prev, ...updates }));
  }, []);

  // 执行保存
  const save = useCallback(async () => {
    if (!isDirty) return;
    setIsSaving(true);
    setError(null);
    try {
      await onSave(draft);
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      onError?.(err);
    } finally {
      setIsSaving(false);
    }
  }, [draft, isDirty, onSave, onSuccess, onError]);

  // 重置回初始状态
  const reset = useCallback(() => {
    setDraft(initialData);
    setError(null);
  }, [initialData]);

  return { draft, isDirty, isSaving, error, updateDraft, save, reset };
}
