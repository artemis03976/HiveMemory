/**
 * 后端以错误响应拒绝了请求：请求已经到达后端，`message` 是可以直接展示的安全文案。
 * 网络失败与流中断不是这个类型，调用方据此区分“后端拒绝”与“连不上后端”。
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly reason?: string;

  constructor(status: number, message: string, code?: string, reason?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.reason = reason;
  }
}

/**
 * 页面由后端提供时，管理凭据随页面以 cookie 下发；后端换了凭据之后，已打开的页面要刷新
 * 才能拿到新的。后端的文案面向所有调用方，这里换成页面上能做的事。
 */
const MANAGEMENT_CREDENTIAL_INVALID = 'request.management_credential_invalid';
const MANAGEMENT_CREDENTIAL_MESSAGE = '管理凭据缺失或已失效，请刷新页面后重试';

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** 旧校验响应可能带凭据或 validator 原文，只使用字段路径和受控类别。 */
function validationMessage(issues: unknown[]): string | undefined {
  const messages: Record<string, string> = {
    missing: '缺少必填字段', extra_forbidden: '不允许此字段',
    string_type: '应提供字符串', int_parsing: '应提供整数', int_type: '应提供整数',
    bool_parsing: '应提供布尔值', bool_type: '应提供布尔值',
    float_parsing: '应提供数字', float_type: '应提供数字',
    list_type: '应提供列表', dict_type: '应提供对象', json_invalid: 'JSON 格式不合法',
    value_error: '字段取值不合法', enum: '取值不在允许范围内',
  };
  const parts = issues.filter(record).map((issue) => {
    const path = Array.isArray(issue.loc)
      ? issue.loc.filter((part) => typeof part === 'string' || typeof part === 'number').join('.')
      : '';
    const message = typeof issue.type === 'string' && Object.hasOwn(messages, issue.type)
      ? messages[issue.type] : '字段取值不合法';
    return path ? `${path}: ${message}` : message;
  });
  return parts.length ? parts.join('；') : undefined;
}

/** API 错误的唯一解析入口；只接受约定结构，未知正文不进入界面。 */
export async function readApiError(response: Response, fallback = '请求失败'): Promise<ApiError> {
  const unknownBody = new ApiError(response.status, `${fallback} (${response.status})`);
  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    // 读取正文时发生取消仍属于取消，不能生成业务错误或触发重试提示。
    if (error instanceof Error && error.name === 'AbortError') throw error;
    return unknownBody;
  }
  if (!record(body)) return unknownBody;
  let message: string;
  if (typeof body.detail === 'string' && body.detail.trim()) {
    message = body.detail;
  } else if (typeof body.message === 'string' && body.message.trim()) {
    message = body.message;
  } else if (record(body.detail) && typeof body.detail.message === 'string' && body.detail.message.trim()) {
    message = body.detail.message;
  } else if (response.status === 422 && Array.isArray(body.detail)) {
    message = validationMessage(body.detail) ?? unknownBody.message;
  } else {
    return unknownBody;
  }
  const code = typeof body.error === 'string' ? body.error : undefined;
  const reason = typeof body.reason === 'string' ? body.reason : undefined;
  if (code === MANAGEMENT_CREDENTIAL_INVALID) message = MANAGEMENT_CREDENTIAL_MESSAGE;
  return new ApiError(response.status, message, code, reason);
}

/** 保留各 API 的默认提示，统一禁止原始 HTML、JSON 和对象字符串化。 */
export async function apiErrorMessage(response: Response, fallback: string): Promise<string> {
  return (await readApiError(response, fallback)).message;
}
