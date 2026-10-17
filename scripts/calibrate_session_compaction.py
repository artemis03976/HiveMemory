"""用固定会话样本比较压缩预算；模型边界使用可审阅的确定性摘要，不调用服务。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, Literal

from hivememory.agent_runtime.context_budget import estimate_context_tokens
from hivememory.agent_runtime.execution.worker import WorkerAgentService
from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
from hivememory.agent_runtime.models import GenerationResult
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.context.view import SessionContextView
from hivememory.config.alice import CompactionConfig, KoakumaConfig
from hivememory.core.models import AgentProfile, ExecutionLabels, TurnEvent
from hivememory.core.protocol.models import AgentRunContext, ConversationTurn, InteractionPayload
from hivememory.prompts.assembler import AgentPromptAssembler


class FixedSummaryWorker(WorkerAgentService):
    """只保留样本首行事实，供人工检查预算和继续执行的材料；不评估模型质量。"""

    def __init__(self) -> None:
        self.requests: list[int] = []

    async def generate_async(
        self, messages: list[dict[str, str]], **options: Any
    ) -> GenerationResult:
        self.requests.append(estimate_context_tokens(messages))
        request = json.loads(messages[1]["content"])
        facts = dict.fromkeys(request["previous_summary"].splitlines())
        for turn in request["turns"]:
            first = turn["messages"][0]["content"].splitlines()[0]
            fact = f"{turn['agent_id']} / {turn['outcome']}: {first}"
            facts.pop(fact, None)
            facts[fact] = None
        return GenerationResult(text="\n".join(tuple(facts)[-8:]).strip())


def fixed_samples() -> dict[str, tuple[ConversationTurn, ...]]:
    """中文代码、需求纠正、工具密集、未完成轮次与多 Agent 五组固定输入。"""
    samples = {}
    for name in ("mixed_code", "correction", "tools", "interrupted", "multi_agent"):
        turns = []
        for index in range(24):
            agent = "reviewer" if name == "multi_agent" and index % 2 else "coder"
            outcome: Literal["completed", "cancelled", "failed"] = (
                ("failed" if index % 2 else "cancelled") if name == "interrupted" else "completed"
            )
            facts = {
                "mixed_code": "保持 parse_time 的 UTC 语义，保留空字符串校验。",
                "correction": (
                    "最新要求：使用 PostgreSQL，不再使用 SQLite。"
                    if index >= 5
                    else "原要求：使用 SQLite。"
                ),
                "tools": "已读取设计资料，需要时重新 READ。",
                "interrupted": "迁移尚未完成；回滚验证仍待执行，不能声称发布成功。",
                "multi_agent": "reviewer 要求保留边界校验；coder 已实现解析但尚未合并。",
            }
            if name == "tools":
                user = facts[name]
                events = [
                    TurnEvent(
                        kind="tool_result",
                        sequence=0,
                        role="user",
                        tool_kind="READ",
                        target=f"design_{index}",
                        status="success",
                        content="资料与代码 abcdefghijklmnop\n" * 650,
                    )
                ]
            else:
                user = (
                    facts[name] + "\n" + ("def parse_time(value): return normalize(value)\n" * 110)
                )
                events = []
            turns.append(
                ConversationTurn(
                    interaction_id=f"{name}-{index}",
                    agent_id=agent,
                    topic_id="sample-topic",
                    created_at=datetime(2026, 10, 17, tzinfo=UTC),
                    payload=InteractionPayload(
                        user_message=user,
                        assistant_final_text="检查记录：" + "trace_result=ok;\n" * 100,
                        turn_events=events,
                        outcome=outcome,
                    ),
                )
            )
        samples[name] = tuple(turns)
    return samples


async def main() -> None:
    """输出请求大小、保留轮数和摘要线索，便于在同一材料上比较默认值。"""
    rows = []
    for label, config in (
        (
            "earlier",
            CompactionConfig(
                trigger_ratio=0.7, target_ratio=0.45, recent_raw_ratio=0.2, summary_max_tokens=512
            ),
        ),
        ("default", CompactionConfig()),
        (
            "later",
            CompactionConfig(
                trigger_ratio=0.8, target_ratio=0.55, recent_raw_ratio=0.3, summary_max_tokens=1536
            ),
        ),
    ):
        input_limit = config.default_context_window - 32768 - config.safety_margin
        for name, turns in fixed_samples().items():
            worker = FixedSummaryWorker()
            context = AgentRunContext(
                labels=ExecutionLabels(agent_id="coder", workspace_id="sample"),
                agent_profile=AgentProfile(persona="分析并继续任务。", language="zh"),
                user_message="继续；先说明还有哪些事项未完成。",
                session_history=turns,
                permitted_mtp_verbs=frozenset({"READ", "RUN"}),
            )
            view = SessionContextView(
                context=context,
                prompt_assembler=AgentPromptAssembler(KoakumaConfig(enabled=False)),
                checkpoint=None,
                input_token_limit=input_limit,
                config=config,
            )
            # 替身遵守 Worker 的文本生成接口，不创建任何网络客户端。
            result = await view.compact(
                summarizer=RelaySummarizer(worker),
                model=ResolvedFrameModel(
                    {"model": "fixed-sample", "max_tokens": 32768}, input_limit
                ),
            )
            rows.append(
                {
                    "profile": label,
                    "sample": name,
                    "triggered": view.should_compact(),
                    "before": result.tokens_before,
                    "after": result.tokens_after,
                    "reduced_turns": result.covered_turns,
                    "summarized_turns": result.summarized_turns,
                    "summary_calls": len(worker.requests),
                    "max_request": max(worker.requests, default=0),
                    "error": type(result.error).__name__ if result.error else None,
                    "summary": result.checkpoint.summary,
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
