/** 失败码来自本轮终态；未知 CPU 的原因保留通用提示，不推测异常正文。 */
export function chatFailureMessage(reason?: string | null): string {
  switch (reason) {
    case 'input_too_large':
      return '本轮输入过大，请缩短消息或减少附件与检索材料。';
    case 'compaction_failed':
      return '较早的对话压缩失败，请重试或新建会话。';
    case 'turn_context_budget_exceeded':
      return '本轮工具结果超出上下文预算，回复未完成，可继续对话或缩小任务。';
    case 'model_context_window_exceeded':
      return '模型拒绝了过长的上下文，请减少输入或选择窗口更大的模型。';
    default:
      return '本轮执行失败';
  }
}
