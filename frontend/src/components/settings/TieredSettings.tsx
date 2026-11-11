import { useState } from 'react';
import { ChevronDown, RotateCcw, Zap } from 'lucide-react';
import { Toggle } from '../common/FormControls';
import { ConfirmDialog } from '../common/ConfirmDialog';
import { readPath, type SettingField, type SettingSectionDef } from '@/constants/settingsSchema';
import type { RegisteredModel } from '@/types/model';

interface TieredSettingsProps {
  sections: SettingSectionDef[];
  config: unknown;
  savedConfig: unknown;
  immediateFields: string[];
  pendingRestart: string[];
  models: RegisteredModel[];
  /** 高级设置在说明下写出字段路径，便于对照配置文件。 */
  showPaths?: boolean;
  updateConfig: (path: string, value: unknown) => void;
  restoreDefaults: (paths: string[]) => Promise<void>;
}

const selectClass =
  'w-56 appearance-none rounded-lg border border-line bg-field py-1.5 pl-3 pr-8 text-sm text-fg focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/50';

function FieldControl({ field, value, models, onChange }: {
  field: SettingField;
  value: unknown;
  models: RegisteredModel[];
  onChange: (value: unknown) => void;
}) {
  if (field.kind === 'switch') {
    return <Toggle checked={value === true} onChange={onChange} />;
  }
  if (field.kind === 'readonly') {
    return <span className="rounded-lg bg-surface-container-high px-2.5 py-1.5 font-mono text-[13px] text-fg-2">{String(value ?? '—')}</span>;
  }
  if (field.kind === 'select' || field.kind === 'model') {
    const current = String(value ?? '');
    const options = field.kind === 'model'
      ? models.map((m) => ({ value: m.id, label: m.display_name ? `${m.id} · ${m.display_name}` : m.id }))
      : field.options ?? [];
    const known = options.some((o) => o.value === current);
    return (
      <span className="relative block">
        <select aria-label={field.label} value={current} onChange={(e) => onChange(e.target.value)} className={selectClass}>
          {!known && <option value={current}>{field.kind === 'model' && models.length > 0 ? `${current}（不在注册表中）` : current}</option>}
          {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-3" aria-hidden="true" />
      </span>
    );
  }
  const scale = field.scale ?? 1;
  const shown = typeof value === 'number' ? Math.round(value * scale * 1000) / 1000 : '';
  return (
    <span className="flex items-center gap-2">
      <input
        type="number"
        aria-label={field.label}
        value={shown}
        min={field.min}
        step={field.step}
        placeholder={field.nullable ? '模型默认' : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === '') {
            if (field.nullable) onChange(null);
            return;
          }
          // 换算过单位的字段（分钟、MB）回写为整数的秒与字节
          onChange(scale === 1 ? Number(raw) : Math.round(Number(raw) / scale));
        }}
        className="w-28 rounded-lg border border-line bg-field px-2.5 py-1.5 text-right text-sm text-fg focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/50"
      />
      {field.unit && <span className="w-10 text-[13px] text-fg-3">{field.unit}</span>}
    </span>
  );
}

/** 常用或高级设置的一页：按小节列出字段，每一项写明保存后何时生效。 */
export function TieredSettings({
  sections, config, savedConfig, immediateFields, pendingRestart, models, showPaths = false, updateConfig, restoreDefaults,
}: TieredSettingsProps) {
  const [confirming, setConfirming] = useState<SettingSectionDef | null>(null);

  return (
    <div className="flex flex-col gap-5">
      {sections.map((section) => {
        const restorable = section.fields.filter((f) => f.kind !== 'readonly');
        return (
          <section key={section.id} className="rounded-2xl border border-line bg-surface-container">
            <div className="flex items-start gap-3 border-b border-line px-5 pb-3 pt-4">
              <div className="flex flex-1 flex-col gap-1">
                <h3 className="flex items-center gap-2 font-manrope text-base font-bold text-fg">
                  {section.title}
                  {section.persona && <span className="rounded-full bg-accent/15 px-2 py-0.5 text-xs font-semibold text-accent">{section.persona}</span>}
                </h3>
                {section.description && <p className="text-[13px] text-fg-3">{section.description}</p>}
              </div>
              {restorable.length > 0 && (
                <button
                  onClick={() => setConfirming(section)}
                  className="flex h-[30px] shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-fg-3 transition-colors hover:bg-ink/5 hover:text-fg"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  恢复本节默认值
                </button>
              )}
            </div>
            {section.fields.map((field) => {
              const value = readPath(config, field.path);
              const dirty = JSON.stringify(value) !== JSON.stringify(readPath(savedConfig, field.path));
              const immediate = immediateFields.includes(field.path);
              return (
                <div
                  key={field.path}
                  className={`flex items-center gap-5 border-t border-line px-5 py-3.5 first-of-type:border-t-0 ${dirty ? 'shadow-[inset_3px_0_0_var(--color-primary)]' : ''}`}
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <span className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-fg">{field.label}</span>
                      {field.kind !== 'readonly' && (
                        <span
                          className="inline-flex items-center gap-1 rounded-full bg-surface-container-high px-2 py-0.5 text-xs text-fg-3"
                          title={immediate ? '保存后马上按新值运行' : '保存后写入配置，重启 HiveMemory 才按新值运行'}
                        >
                          {immediate ? <Zap className="h-3 w-3" /> : <RotateCcw className="h-3 w-3" />}
                          {immediate ? '立即生效' : '重启后生效'}
                        </span>
                      )}
                      {pendingRestart.includes(field.path) && !dirty && (
                        <span className="inline-flex items-center gap-1.5 rounded-full bg-magic-metal/15 px-2 py-0.5 text-xs font-semibold text-magic-metal">
                          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-magic-metal" />
                          等待重启
                        </span>
                      )}
                      {dirty && <span className="rounded-full bg-primary/15 px-2 py-0.5 text-xs font-semibold text-primary">已修改</span>}
                    </span>
                    <span className="text-[13px] leading-normal text-fg-3">{field.description}</span>
                    {showPaths && <span className="font-mono text-xs text-fg-3">{field.path}</span>}
                  </div>
                  <FieldControl field={field} value={value} models={models} onChange={(v) => updateConfig(field.path, v)} />
                </div>
              );
            })}
          </section>
        );
      })}

      <ConfirmDialog
        isOpen={confirming !== null}
        title={confirming ? `恢复“${confirming.title}”的默认值？` : ''}
        message="只影响这一节，其他设置不变。恢复后仍需保存才会生效。"
        confirmText="恢复"
        onConfirm={() => {
          if (confirming) void restoreDefaults(confirming.fields.filter((f) => f.kind !== 'readonly').map((f) => f.path));
          setConfirming(null);
        }}
        onCancel={() => setConfirming(null)}
      />
    </div>
  );
}
