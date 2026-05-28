import { useState, useCallback, useMemo, useEffect } from 'react';

function deepEqual(obj1: unknown, obj2: unknown): boolean {
  if (obj1 === obj2) return true;

  if (typeof obj1 !== 'object' || typeof obj2 !== 'object' || obj1 == null || obj2 == null) {
    return false;
  }

  const record1 = obj1 as Record<string, unknown>;
  const record2 = obj2 as Record<string, unknown>;
  const keys1 = Object.keys(record1);
  const keys2 = Object.keys(record2);

  if (keys1.length !== keys2.length) return false;

  for (const key of keys1) {
    if (!keys2.includes(key)) return false;
    if (!deepEqual(record1[key], record2[key])) return false;
  }

  return true;
}

interface SubmitOptions {
  force?: boolean;
}

interface UseDraftOptions<T, TResult = void> {
  initialData: T;
  onSave: (draftData: T) => Promise<TResult>;
  onSuccess?: (result: TResult, draftData: T) => void;
  onError?: (error: unknown) => void;
  skipUnchangedSubmit?: boolean;
}

export function useDraft<T, TResult = void>({
  initialData,
  onSave,
  onSuccess,
  onError,
  skipUnchangedSubmit = true,
}: UseDraftOptions<T, TResult>) {
  const [draft, setDraft] = useState<T>(initialData);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setDraft((prevDraft) => (deepEqual(prevDraft, initialData) ? prevDraft : initialData));
  }, [initialData]);

  const isDirty = useMemo(() => !deepEqual(initialData, draft), [initialData, draft]);

  const updateDraft = useCallback((updates: Partial<T>) => {
    setDraft(prev => ({ ...prev, ...updates }));
  }, []);

  const replaceDraft = useCallback((nextDraft: T) => {
    setDraft(nextDraft);
  }, []);

  const submit = useCallback(async (options: SubmitOptions = {}) => {
    if (skipUnchangedSubmit && !options.force && !isDirty) return undefined;

    setIsSaving(true);
    setError(null);

    try {
      const result = await onSave(draft);
      onSuccess?.(result, draft);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      onError?.(err);
      return undefined;
    } finally {
      setIsSaving(false);
    }
  }, [draft, isDirty, onSave, onSuccess, onError, skipUnchangedSubmit]);

  const reset = useCallback(() => {
    setDraft(initialData);
    setError(null);
  }, [initialData]);

  const save = useCallback(() => submit(), [submit]);

  return {
    draft,
    isDirty,
    isSaving,
    error,
    updateDraft,
    replaceDraft,
    submit,
    save,
    reset,
  };
}
