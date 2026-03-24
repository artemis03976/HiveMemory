export function SettingSection({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <section className="mb-8">
      <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">{title}</h2>
      <div className="bg-surface-container-low border border-white/5 rounded-2xl overflow-hidden ghost-border">
        {children}
      </div>
    </section>
  );
}

export function SettingRow({ label, description, children }: { label: string, description?: string, children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between p-5 border-b border-white/5 last:border-0 hover:bg-white/2 transition-colors">
      <div className="pr-8">
        <div className="text-sm font-medium text-slate-200">{label}</div>
        {description && <div className="text-xs text-slate-500 mt-1 leading-relaxed">{description}</div>}
      </div>
      <div className="shrink-0">
        {children}
      </div>
    </div>
  );
}

export function Toggle({ checked, onChange, disabled }: { checked: boolean, onChange?: (c: boolean) => void, disabled?: boolean }) {
  return (
    <button 
      disabled={disabled}
      onClick={() => onChange?.(!checked)}
      className={`relative w-11 h-6 rounded-full transition-colors ${checked ? 'bg-primary' : 'bg-white/10'} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
    </button>
  );
}

export function Input({ type = "text", value, onChange, placeholder, className = "w-48", step, disabled, error }: any) {
  return (
    <div className="flex flex-col gap-1 items-end">
      <input 
        type={type} 
        value={value ?? ''}
        onChange={(e) => onChange?.(type === 'number' ? Number(e.target.value) : e.target.value)}
        placeholder={placeholder}
        step={step}
        disabled={disabled}
        className={`bg-black/20 border ${error ? 'border-red-500/50 focus:ring-red-500/50 focus:border-red-500/50' : 'border-white/10 focus:ring-primary/50 focus:border-primary/50'} rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 transition-all font-mono ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
      />
      {error && <span className="text-xs text-red-400 max-w-[200px] text-right">{error}</span>}
    </div>
  );
}

export function Select({ options, value, onChange, error }: { options: {label: string, value: string}[], value?: string, onChange?: (v: string) => void, error?: string }) {
  return (
    <div className="flex flex-col gap-1 items-end">
      <select 
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className={`bg-black/20 border ${error ? 'border-red-500/50 focus:ring-red-500/50 focus:border-red-500/50' : 'border-white/10 focus:ring-primary/50 focus:border-primary/50'} rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 transition-all cursor-pointer font-mono outline-none`}
      >
        {options.map(o => <option key={o.value} value={o.value} className="bg-surface-container">{o.label}</option>)}
      </select>
      {error && <span className="text-xs text-red-400 max-w-[200px] text-right">{error}</span>}
    </div>
  );
}
