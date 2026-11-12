import { useCallback, useEffect, useState } from 'react';
import { Check, Copy, Folder, Loader2, RefreshCw } from 'lucide-react';

import { ConfirmDialog } from '../common/ConfirmDialog';
import { useToastStore } from '@/stores';
import { fetchPlugins, installPlugin, uninstallPlugin } from '@/services/pluginApi';
import type { HarnessId, PluginActionResult, PluginFailure, PluginList, PluginStatus } from '@/types/plugin';
import { pluginCardState, pluginFailureMessage, type PluginCardState } from '@/utils/pluginStatus';

/** 每个 harness 的简介与装好之后要做的事；名称、命令与凭据变量由后端给出。 */
const HARNESS_COPY: Record<HarnessId, { initials: string; desc: string; next: string[] }> = {
  'claude-code': {
    initials: 'CC',
    desc: '四个记忆工具（检索、读取、写入、更新）与对话回流，附带使用手册 skill。',
    next: ['完全退出 Claude Code（包括已经打开的终端）后重新打开，记忆工具就会出现。'],
  },
  codex: {
    initials: 'CX',
    desc: '四个记忆工具与对话回流。Codex 要求在它自己的界面中信任插件的 hook。',
    next: [
      '完全退出 Codex（桌面应用从托盘退出）后重新打开。',
      '打开 Codex 时会提示 Hooks need review：审阅并信任 hivememory-connector 的五个 hook，否则对话不会回流。',
    ],
  },
  pi: {
    initials: 'pi',
    desc: '扩展注册的四个记忆工具与对话回流，附带使用手册 skill。升级 HiveMemory 后直接读到新版本。',
    next: ['在 pi 中执行 /reload，或重新打开 pi。'],
  },
  dsh: {
    initials: 'DS',
    desc: 'dsh 桌面端的插件：四个记忆工具与对话回流，附带使用手册 skill。只有 dsh 桌面端退出时才能安装。',
    next: ['重新打开 dsh 桌面端；在消息框输入 /hive，Skills 列表中出现 hivememory 即已加载。'],
  },
};

const BADGES: Record<PluginCardState, { label: string; cls: string }> = {
  installed: { label: '已安装', cls: 'bg-magic-wood/12 text-magic-wood' },
  needs_update: { label: '需要更新', cls: 'bg-magic-metal/12 text-magic-metal' },
  not_installed: { label: '未安装', cls: 'bg-ink/5 text-fg-3' },
  missing: { label: '没有找到', cls: 'bg-magic-metal/12 text-magic-metal' },
  unavailable: { label: '不能代为安装', cls: 'bg-ink/5 text-fg-3' },
};

type Busy = 'installing' | 'removing';

interface CardResult {
  failure: PluginFailure | null;
  /** 刚刚完成的动作：安装成功后显示下一步，卸载后提示 harness 一侧是否还留着插件。 */
  action: 'installed' | 'removed';
}

const primaryCls =
  'h-9 shrink-0 rounded-xl bg-brand px-4 text-sm font-semibold text-white shadow-glow disabled:cursor-not-allowed disabled:opacity-60';
const secondaryCls =
  'h-9 shrink-0 rounded-xl border border-line-strong bg-surface-container-low px-3 text-[13px] font-semibold text-fg hover:bg-surface-container-high disabled:cursor-not-allowed disabled:opacity-60';

function detailText(status: PluginStatus, state: PluginCardState, version: string): string {
  const mode = status.install_mode === 'copy' ? '复制安装' : '链接到插件目录';
  switch (state) {
    case 'installed':
    case 'needs_update': {
      const agents = status.allowed_agent_ids ? status.allowed_agent_ids.join('、') : '任意 Agent';
      const base = `可声明的 Agent：${agents} · 凭据在 ${status.credential_env} · ${mode}`;
      return state === 'needs_update' ? `${base} · HiveMemory 已升级到 ${version}，插件还没有更新` : base;
    }
    case 'not_installed':
      return `已找到 ${status.command}`;
    case 'missing':
      return status.harness === 'dsh' ? '没有找到 dsh 桌面端' : `没有在 PATH 上找到 ${status.name}`;
    case 'unavailable':
      return '插件目录不可用，或者 HiveMemory 不是在 Windows 上运行';
  }
}

function CommandBlock({ status, intro }: { status: PluginStatus; intro: string }) {
  const { addToast } = useToastStore();
  const text = status.manual_commands.join('\n');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      addToast('命令已复制', 'success', 1600);
    } catch {
      addToast('复制失败，请手动选中复制', 'error');
    }
  };
  return (
    <div className="flex flex-col gap-2">
      <span className="text-[13px] text-fg-2">{intro}</span>
      <div className="relative">
        <pre className="whitespace-pre-wrap break-all rounded-xl border border-line bg-field py-3 pl-3 pr-24 font-mono text-xs leading-relaxed text-fg">
          {text}
        </pre>
        <button
          onClick={() => void copy()}
          aria-label="复制命令"
          className="absolute right-2 top-2 flex h-7 items-center gap-1 rounded-lg border border-line-strong bg-surface-container-low px-2.5 text-xs font-semibold text-fg hover:bg-surface-container-high"
        >
          <Copy className="h-3 w-3" aria-hidden="true" />
          复制
        </button>
      </div>
    </div>
  );
}

function PluginCard({
  status,
  version,
  busy,
  result,
  locked,
  onInstall,
  onRegisterManual,
  onUninstall,
  onRecheck,
}: {
  status: PluginStatus;
  version: string;
  busy: Busy | undefined;
  result: CardResult | undefined;
  locked: boolean;
  onInstall: () => void;
  onRegisterManual: () => void;
  onUninstall: () => void;
  onRecheck: () => void;
}) {
  const [showCommands, setShowCommands] = useState(false);
  const copy = HARNESS_COPY[status.harness];
  const state = pluginCardState(status);
  const badge = busy
    ? { label: busy === 'installing' ? '安装中' : '卸载中', cls: 'bg-magic-water/12 text-magic-water' }
    : BADGES[state];
  const failure = result?.failure ?? null;
  const justInstalled = result?.action === 'installed' && failure === null && status.installed;
  const installed = state === 'installed' || state === 'needs_update';
  const border = failure ? 'border-magic-fire/60' : justInstalled ? 'border-magic-wood/60' : 'border-line';
  // 插件命令装不上时，用户可以照命令手动安装，再在这里完成登记：凭据仍由 HiveMemory 交付。
  const canRegisterManually = status.available && !installed;

  return (
    <section aria-label={status.name} className={`rounded-2xl border bg-surface-container ${border}`}>
      <div className="flex items-start gap-3.5 p-4">
        <span
          aria-hidden="true"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-line-strong bg-surface-container-high font-manrope text-sm font-extrabold text-fg-2"
        >
          {copy.initials}
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-manrope text-base font-bold text-fg">{status.name}</h3>
            <span className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-semibold ${badge.cls}`}>
              {busy ? <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" /> : <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />}
              {badge.label}
            </span>
          </div>
          <p className="text-[13px] leading-relaxed text-fg-2">{copy.desc}</p>
          <p className="break-all font-mono text-xs leading-relaxed text-fg-3">{detailText(status, state, version)}</p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {(state === 'not_installed' || (failure && !installed && status.available)) && (
            <button onClick={onInstall} disabled={locked} className={primaryCls}>
              {failure ? '重试' : '安装'}
            </button>
          )}
          {state === 'missing' && !failure && (
            <>
              <button onClick={() => setShowCommands((v) => !v)} className={secondaryCls}>
                {showCommands ? '收起命令' : '手动安装的命令'}
              </button>
              <button onClick={onRecheck} disabled={locked} className={secondaryCls}>重新检查</button>
            </>
          )}
          {installed && (
            <>
              {status.available && (
                <button onClick={onInstall} disabled={locked} className={state === 'needs_update' ? primaryCls : secondaryCls}>
                  {state === 'needs_update' ? '更新' : '重新安装'}
                </button>
              )}
              <button onClick={onUninstall} disabled={locked} className={`${secondaryCls} text-magic-fire`}>卸载</button>
            </>
          )}
        </div>
      </div>

      {justInstalled && (
        <div role="status" className="mb-4 ml-[70px] mr-4 flex items-start gap-2.5 rounded-xl bg-magic-wood/10 px-3.5 py-2.5">
          <Check className="mt-0.5 h-4 w-4 shrink-0 text-magic-wood" aria-hidden="true" />
          <div className="flex flex-col gap-1 text-[13px] leading-relaxed">
            <b className="font-semibold text-fg">已安装，还差一步</b>
            {copy.next.map((line) => <span key={line} className="text-fg-2">{line}</span>)}
          </div>
        </div>
      )}

      {failure && (
        <div role="alert" className="mb-4 ml-[70px] mr-4 flex flex-col gap-2.5 rounded-xl bg-magic-fire/8 px-3.5 py-3">
          <span className="text-[13px] leading-relaxed text-fg">
            <b className="font-semibold text-magic-fire">{result?.action === 'removed' ? '接入已删除，但插件没有卸载：' : '没有安装：'}</b>
            {pluginFailureMessage(failure, status, result?.action ?? 'installed')}
          </span>
          {failure.output_tail.length > 0 && (
            <>
              <span className="text-xs text-fg-3">命令输出的最后几行{failure.command ? `（${failure.command}）` : ''}</span>
              <pre className="whitespace-pre-wrap break-all rounded-xl border border-line bg-field px-3 py-2.5 font-mono text-xs leading-relaxed text-fg-2">
                {failure.output_tail.join('\n')}
              </pre>
            </>
          )}
          {result?.action === 'installed' && status.manual_commands.length > 0 && (
            <CommandBlock status={status} intro="手动安装插件的命令：" />
          )}
          {canRegisterManually && result?.action === 'installed' && (
            <div className="flex items-center justify-end">
              <button onClick={onRegisterManual} disabled={locked} className={secondaryCls}>已用命令装好，完成登记</button>
            </div>
          )}
        </div>
      )}

      {!failure && (state === 'unavailable' || (state === 'missing' && showCommands)) && (
        <div className="mb-4 ml-[70px] mr-4 flex flex-col gap-2.5">
          <CommandBlock
            status={status}
            intro={state === 'unavailable' ? '这里不能代为安装，可以在装有该 harness 的电脑上手动安装：' : `装好 ${status.name} 后点“重新检查”；也可以手动安装：`}
          />
          {canRegisterManually && (
            <div className="flex items-center justify-end">
              <button onClick={onRegisterManual} disabled={locked} className={secondaryCls}>已用命令装好，完成登记</button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/** 插件：为外部 harness 安装、卸载 HiveMemory 插件。新安装的 HiveMemory 不连接任何 harness。 */
export function PluginSettings({ onInstalledCount }: { onInstalledCount?: (count: number) => void }) {
  const { addToast } = useToastStore();
  const [list, setList] = useState<PluginList | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<Partial<Record<HarnessId, Busy>>>({});
  const [results, setResults] = useState<Partial<Record<HarnessId, CardResult>>>({});
  const [confirming, setConfirming] = useState<PluginStatus | null>(null);

  const publish = useCallback((next: PluginList) => {
    setList(next);
    onInstalledCount?.(next.harnesses.filter((h) => h.installed).length);
  }, [onInstalledCount]);

  const load = useCallback(async () => {
    try {
      publish(await fetchPlugins());
    } catch (err) {
      addToast('读取插件状态失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast, publish]);

  useEffect(() => { void load(); }, [load]);

  const apply = (harness: HarnessId, result: PluginActionResult, action: CardResult['action']) => {
    setResults((r) => ({ ...r, [harness]: { failure: result.failure, action } }));
    // 安装与卸载同一时刻只做一个，这里的列表就是发起这次动作时的最新状态。
    if (list) publish({ ...list, harnesses: list.harnesses.map((h) => (h.harness === harness ? result.status : h)) });
  };

  const act = async (status: PluginStatus, kind: Busy, request: () => Promise<PluginActionResult>) => {
    setBusy((b) => ({ ...b, [status.harness]: kind }));
    setResults((r) => ({ ...r, [status.harness]: undefined }));
    try {
      const result = await request();
      apply(status.harness, result, kind === 'installing' ? 'installed' : 'removed');
      if (kind === 'removing' && !result.failure) addToast(`${status.name} 的插件已卸载`, 'success', 1800);
    } catch (err) {
      addToast((err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setBusy((b) => ({ ...b, [status.harness]: undefined }));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-primary">
        <RefreshCw className="h-6 w-6 animate-spin" />
      </div>
    );
  }
  if (!list) return <p className="py-20 text-center text-fg-3">插件状态暂不可用</p>;

  // 安装与卸载都会改动 harness 的配置与接入登记，同一时刻只做一个。
  const locked = Object.values(busy).some(Boolean);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="font-manrope text-[22px] font-extrabold text-fg">插件</h2>
        <p className="mt-1 text-sm leading-relaxed text-fg-2">
          为外部 harness 安装插件后，它的 Agent 才能检索、读取、写入与更新记忆，结束的对话也会回流到 HiveMemory。新安装的 HiveMemory 不连接任何 harness。
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-line bg-surface-container px-3.5 py-2.5 text-[13px] text-fg-2">
        <span className="flex items-center gap-1.5"><Folder className="h-3.5 w-3.5 text-fg-3" aria-hidden="true" />插件目录</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg-3" title={list.integrations_dir}>{list.integrations_dir}</span>
        <span className="rounded-full bg-ink/5 px-2 py-0.5 text-xs text-fg-3">与后端同为 {list.version}</span>
      </div>

      {list.harnesses.map((status) => (
        <PluginCard
          key={status.harness}
          status={status}
          version={list.version}
          busy={busy[status.harness]}
          result={results[status.harness]}
          locked={locked}
          onInstall={() => void act(status, 'installing', () => installPlugin(status.harness))}
          onRegisterManual={() => void act(status, 'installing', () => installPlugin(status.harness, false))}
          onUninstall={() => setConfirming(status)}
          onRecheck={() => { setLoading(true); void load(); }}
        />
      ))}

      <p className="text-xs leading-relaxed text-fg-3">
        安装时，凭据写入 Windows 用户环境变量。已经打开的 harness 和终端读不到新变量，需要完全退出后重新打开。这台电脑上同一用户运行的程序都能读到这个凭据，它只用来防止误用。
      </p>

      <ConfirmDialog
        isOpen={confirming !== null}
        title={confirming ? `卸载 ${confirming.name} 的插件？` : ''}
        message="会卸载 harness 中的插件，删除它的接入登记、凭据与用户环境变量。之后这个 harness 不能再访问记忆；Agent 的操作权限和已经写入的记忆都保留。"
        confirmText="卸载"
        onConfirm={() => {
          const target = confirming;
          setConfirming(null);
          if (target) void act(target, 'removing', () => uninstallPlugin(target.harness));
        }}
        onCancel={() => setConfirming(null)}
      />
    </div>
  );
}
