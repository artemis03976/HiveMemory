import type { RuntimeEvent } from '@/types/workspace';
import type { EventDescription, EventTone, GatewayStep } from '@/types/runs';
import { GENERATION_SOURCE_LABEL, SETTLE_REASON_LABEL, shortId } from '@/stores/runs/labels';

const CHAT_PHASE_LABEL: Record<string, string> = {
  created: '创建',
  preparing: '准备阶段',
  streaming: '执行阶段',
  finalizing: '收尾阶段',
};

const INGRESS_LABEL: Record<string, string> = {
  active_chat: '主动对话',
  passive_memory: '被动记忆',
};

const RETRIEVAL_LABEL: Record<string, string> = {
  SKIP: '不检索',
  DENSE: '稠密检索',
  SPARSE: '稀疏检索',
  HYBRID: '混合检索',
};

const GATEWAY_STEP_LABEL: Record<string, string> = {
  entry_interception: '入口拦截',
  candidate_topics_preparation: '候选话题',
  topic_routing: '话题路由',
  routed_topic_preparation: '准备路由话题',
  simple_chat_defaults: '默认值',
  user_query_analysis: '查询分析',
};

function str(event: RuntimeEvent, field: string): string | null {
  const value = event.data?.[field];
  return typeof value === 'string' && value ? value : null;
}

function num(event: RuntimeEvent, field: string): number | null {
  const value = event.data?.[field];
  return typeof value === 'number' ? value : null;
}

/** 毫秒数显示为 `812ms`、`1.36s`、`18.4s` 或 `2m05s`。 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.floor((ms % 60_000) / 1000);
  return `${minutes}m${seconds.toString().padStart(2, '0')}s`;
}

function joined(...parts: (string | null | false | undefined)[]): string | null {
  const present = parts.filter((part): part is string => Boolean(part));
  return present.length > 0 ? present.join(' · ') : null;
}

function describe(text: string, detail: string | null = null, tone: EventTone = 'normal'): EventDescription {
  return { text, detail, tone };
}

function describeActorOperation(event: RuntimeEvent): EventDescription {
  const operation = str(event, 'operation');
  const entry = str(event, 'entry');
  const duration = num(event, 'duration_ms');
  const context = joined(entry && `入口 ${entry}`, duration !== null && `耗时 ${formatDuration(duration)}`);
  const name: Record<string, string> = {
    'memory.search': '检索',
    'memory.read': '读取',
    'memory.run': 'RUN',
    'memory.write': '写入',
    'memory.update': '更新',
    'agent.call': 'CALL',
  };
  const label = (operation && name[operation]) ?? operation ?? '操作';

  if (event.status === 'rejected') return describe(`${label}被拒：${event.reason ?? '未知原因'}`, context, 'warn');
  if (event.status === 'failed') {
    return describe(`${label}失败：${event.reason ?? '未知原因'}`, joined(str(event, 'error_class'), context), 'error');
  }

  switch (operation) {
    case 'memory.search':
      return describe(`检索 ${num(event, 'result_count') ?? 0} 条`, context);
    case 'memory.read':
      return describe(`读取 ${num(event, 'delivered_count') ?? 0}/${num(event, 'requested_count') ?? 0}`, context);
    case 'memory.run':
      return describe(`RUN ${str(event, 'resolution') ?? ''}`.trim(), context);
    case 'memory.write':
      return describe('写入已接纳', joined(str(event, 'pending_alias'), context), 'ok');
    case 'memory.update':
      return describe('更新已接纳', joined(str(event, 'pending_alias'), context), 'ok');
    case 'agent.call':
      return describe(`CALL ${str(event, 'callee_agent_id') ?? ''}`.trim(), context);
    default:
      return describe(label, context);
  }
}

function describeGatewayCompleted(event: RuntimeEvent): EventDescription {
  const steps = num(event, 'completed_steps');
  if (str(event, 'outcome_kind') === 'command') {
    return describe(`Gateway 完成：命令 ${str(event, 'command_id') ?? ''}`.trim(), steps !== null ? `${steps} 步` : null);
  }
  const topic = str(event, 'topic_id');
  const retrieval = str(event, 'retrieval_mode');
  return describe(
    `Gateway 完成：${topic ? '继续已有话题' : '路由到新话题'}`,
    joined(
      topic && `话题 ${shortId(topic)}`,
      str(event, 'intent_type') && `意图 ${str(event, 'intent_type')}`,
      retrieval && (RETRIEVAL_LABEL[retrieval] ?? retrieval),
      steps !== null && `${steps} 步`,
    ),
  );
}

/** 时间线上一条事件的描述：每个事件类型一句话，由字段拼出，不显示正文；未知类型回退为类型名。 */
export function describeEvent(event: RuntimeEvent): EventDescription {
  const reason = event.reason ?? '未知原因';
  const errorClass = str(event, 'error_class');
  switch (event.event_type) {
    case 'chat.run.created':
      return describe('对话开始', `进程 ${shortId(event.process_id)}`, 'key');
    case 'chat.run.status':
      return describe(`进入${CHAT_PHASE_LABEL[event.status ?? ''] ?? event.status ?? '下一阶段'}`);
    case 'chat.run.cancel_requested':
      return describe('请求停止', joined(event.status && `结果 ${event.status}`, event.reason));
    case 'chat.run.completed':
      return describe('对话完成', event.topic_id ? `话题 ${shortId(event.topic_id)}` : null, 'ok');
    case 'chat.run.cancelled':
      return describe(`已停止（${event.reason ?? str(event, 'close_reason') ?? str(event, 'phase') ?? '未知原因'}）`);
    case 'chat.run.failed':
      return describe(`对话失败：${reason}`, errorClass, 'error');

    case 'agent.run.started':
      return describe('Agent 开始执行', event.agent_id);
    case 'agent.run.completed':
      return describe(`Agent 完成（MTP ${num(event, 'mtp_iterations') ?? 0} 轮）`);
    case 'agent.run.cancelled':
      return describe('Agent 停止', str(event, 'close_reason'));
    case 'agent.run.failed':
      return describe(`Agent 失败：${reason}`, errorClass, 'error');
    case 'agent.context.compaction': {
      if (event.status === 'started') return describe('上下文压缩开始');
      const before = num(event, 'tokens_before');
      const after = num(event, 'tokens_after');
      return describe(`上下文压缩 ${before ?? '?'}→${after ?? '?'} tokens`, event.reason);
    }
    case 'agent.context.budget_exceeded':
      return describe(
        '超出上下文预算',
        `${num(event, 'estimated_tokens') ?? '?'} / ${num(event, 'input_token_limit') ?? '?'} tokens`,
        'warn',
      );

    case 'gateway.workflow.started': {
      const ingress = str(event, 'ingress_mode');
      return describe(`Gateway 开始分析（${ingress ? (INGRESS_LABEL[ingress] ?? ingress) : '未知入口'}）`);
    }
    case 'gateway.workflow.completed':
      return describeGatewayCompleted(event);
    case 'gateway.workflow.failed':
      return describe(
        `Gateway 失败：${reason}`,
        joined(str(event, 'step_id') && `出错的步骤 ${str(event, 'step_id')}`, errorClass),
        'error',
      );
    case 'gateway.analysis.capability.completed': {
      const duration = num(event, 'duration_ms');
      const durationText = duration !== null ? formatDuration(duration) : null;
      return str(event, 'error')
        ? describe('查询分析失败', joined(str(event, 'error'), durationText), 'warn')
        : describe('查询分析完成', durationText);
    }

    case 'actor.operation':
      return describeActorOperation(event);

    case 'topic.created':
      return describe(
        `建立话题（${str(event, 'created_by') === 'prepare' ? '为写入预先确定' : '应用交互时'}）`,
        `话题 ${shortId(event.topic_id)}`,
        'key',
      );
    case 'topic.settled': {
      const settleReason = str(event, 'reason');
      return describe(
        `话题结算（${settleReason ? (SETTLE_REASON_LABEL[settleReason] ?? settleReason) : '未知原因'}）`,
        event.data?.generation_submitted === false ? '没有可保存的材料' : '已提交生成',
        'key',
      );
    }
    case 'topic.deleted':
      return describe('话题删除', `话题 ${shortId(event.topic_id)}`, 'key');

    case 'import.turn.accepted':
      return describe(
        `导入接纳（${event.data?.is_new_topic === true ? '新话题' : '已有话题'}）`,
        joined(event.topic_id && `话题 ${shortId(event.topic_id)}`, num(event, 'event_count') !== null && `${num(event, 'event_count')} 个事件`),
        'ok',
      );
    case 'import.turn.duplicate':
      return describe('重复，已忽略', event.reason);
    case 'import.turn.skipped':
      return describe('没有输出，跳过', event.reason);
    case 'import.turn.deferred':
      return describe(`推迟：${reason}`, joined(errorClass, '连接器保留这一轮，稍后重发'), 'warn');

    case 'memory.task.created': {
      const source = str(event, 'source');
      return describe(
        `生成任务创建（${source ? (GENERATION_SOURCE_LABEL[source] ?? source) : '未知来源'}）`,
        str(event, 'pending_alias') ?? str(event, 'label'),
        'key',
      );
    }
    case 'memory.task.status':
      return describe(event.status === 'running' ? '开始生成' : `状态 ${event.status ?? '未知'}`);
    case 'memory.task.cancel_requested':
      return describe('请求取消', event.reason);
    case 'memory.task.completed':
      return describe(`生成完成 → ${str(event, 'canonical_alias') ?? '未知记忆'}`, null, 'ok');
    case 'memory.task.cancelled':
      return describe(`已取消（${reason}）`);
    case 'memory.task.failed':
      return describe(`生成失败：${reason}`, errorClass, 'error');

    case 'work.queued':
      return describe('入队', event.task_id);
    case 'work.started':
      return describe('开始执行', num(event, 'attempt_count') !== null ? `第 ${num(event, 'attempt_count')} 次` : null);
    case 'work.retry_scheduled':
      return describe('重试排队', event.reason, 'warn');
    case 'work.succeeded':
      return describe('执行成功');
    case 'work.failed':
      return describe(`执行失败：${reason}`, null, 'error');
    case 'work.dead_lettered':
      return describe(`进入死信：${reason}`, null, 'error');
    case 'work.cancel_requested':
      return describe('请求取消', event.reason);
    case 'work.cancelled':
      return describe('已取消', event.reason);
    case 'work.rejected':
      return describe(`拒绝：${reason}`, event.task_id, 'warn');

    case 'maintenance.task.completed': {
      const duration = num(event, 'duration_ms');
      return describe(
        '维护完成',
        joined(num(event, 'run_count') !== null && `第 ${num(event, 'run_count')} 次`, duration !== null && formatDuration(duration)),
        'ok',
      );
    }
    case 'maintenance.task.failed':
      return describe(`维护失败：${reason}`, errorClass, 'error');

    case 'system.starting':
      return describe('系统启动中', null, 'key');
    case 'system.ready':
      return describe('系统就绪', stepsDone(event), 'ok');
    case 'system.start_failed':
      return describe(`启动失败：${str(event, 'failed_step') ?? reason}`, errorClass, 'error');
    case 'system.shutting_down':
      return describe('系统关闭中', null, 'key');
    case 'system.stopped':
      return describe('系统已关闭', stepsDone(event), 'ok');
    case 'system.stop_failed':
      return describe(`关闭失败：${str(event, 'failed_step') ?? reason}`, errorClass, 'error');

    case 'subsystem.operation.started':
      return describe(`${str(event, 'operation_name') ?? '子系统操作'}开始`);
    case 'subsystem.operation.completed': {
      const duration = num(event, 'duration_ms');
      return describe(
        `${str(event, 'operation_name') ?? '子系统操作'}完成（${event.status ?? 'completed'}）`,
        duration !== null ? formatDuration(duration) : null,
        event.status === 'completed' ? 'ok' : 'warn',
      );
    }
    case 'subsystem.operation.failed':
      return describe(`${str(event, 'operation_name') ?? '子系统操作'}失败`, errorClass, 'error');

    case 'event.stream.gap':
      return describe('事件流断档', num(event, 'dropped_count') !== null ? `丢失 ${num(event, 'dropped_count')} 条` : null, 'warn');

    default:
      return describe(event.event_type);
  }
}

function stepsDone(event: RuntimeEvent): string | null {
  const completed = event.data?.completed_steps;
  const duration = num(event, 'duration_ms');
  return joined(Array.isArray(completed) && `完成 ${completed.length} 步`, duration !== null && formatDuration(duration));
}

/** Gateway 终态事件中按执行顺序的各步耗时。 */
export function gatewaySteps(event: RuntimeEvent): GatewayStep[] {
  const durations = event.data?.step_durations_ms;
  if (!durations || typeof durations !== 'object' || Array.isArray(durations)) return [];
  return Object.entries(durations as Record<string, unknown>)
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number')
    .map(([stepId, durationMs]) => ({ stepId, label: GATEWAY_STEP_LABEL[stepId] ?? stepId, durationMs }));
}

/** Gateway 终态事件中发生过的降级，按"步骤：原因"显示。 */
export function gatewayFallbacks(event: RuntimeEvent): string[] {
  const fallbacks = event.data?.fallbacks;
  if (!Array.isArray(fallbacks)) return [];
  return fallbacks.flatMap((item) => {
    if (!item || typeof item !== 'object') return [];
    const { step_id: stepId, reason } = item as { step_id?: unknown; reason?: unknown };
    if (typeof stepId !== 'string') return [];
    return [`${GATEWAY_STEP_LABEL[stepId] ?? stepId}：${typeof reason === 'string' ? reason : '未知原因'}`];
  });
}
