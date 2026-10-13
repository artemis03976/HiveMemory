"""
子代理调用 E2E 测试 (真实 LLM)

目标：
    1. 验证真实对话下 CALL -> 子代理执行 -> CALL response 回填链路
    2. 验证流式事件契约：统一 token/mtp* + scope 命名空间
    3. 验证星型拓扑约束：子代理侧不应继续发起 CALL
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.e2e.conftest import root_access, workspace_for
from tests.helpers.registered_chat import stream_registered_chat

logger = logging.getLogger(__name__)
# 系统由 session 级 fixture 在 session 事件循环中启动，测试须运行在同一循环。
pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.asyncio(loop_scope="session")]


def _event_summary(events: list[dict[str, Any]]) -> str:
    """压缩事件轨迹，便于失败时快速诊断。"""
    rows: list[str] = []
    for e in events:
        event = e.get("event")
        data = e.get("data", {}) or {}
        scope = data.get("scope", "-")
        verb = data.get("verb", "-")
        status = data.get("status", "-")
        rows.append(f"{event}(scope={scope}, verb={verb}, status={status})")
    return " -> ".join(rows)


def _commands(events: list[dict[str, Any]]) -> list[str]:
    """动作来自公开 mtp_start，封口用的 turn_events 不进入 done wire。"""
    return [
        event["data"]["verb"]
        for event in events
        if event["event"] == "mtp_start" and event["data"]["scope"] == "main"
    ]


async def _caller_with_coder_doll(system, e2e_user) -> str:
    """取一个新的测试用户，并在其 Workspace 中经 root 管理入口创建 coder_doll。

    CALL 目标按调用方的 Workspace 解析，Profile 必须与调用方同属一个 Workspace。
    """
    user_id = e2e_user()
    workspace = workspace_for(user_id)
    async with root_access(system, workspace) as access:
        await system.root_agent_service.create_agent_profile(
            target_workspace=workspace,
            title="Coder Doll",
            alias="coder_doll",
            summary="专用于代码生成的子代理",
            content=(
                "你是 coder_doll，擅长编写简洁、正确、可读的 Python 代码。"
                "在收到任务时优先直接给出可运行实现。"
            ),
            tags=["agent", "profile", "coder"],
            agent_config={"model_name": "default", "temperature": 0.2, "language": "zh"},
            access=access,
        )
    return user_id


async def _collect_stream_events(
    system,
    user_message: str,
    user_id: str,
    agent_id: str = "omni_doll",
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for event in stream_registered_chat(
        system,
        message=user_message,
        user_id=user_id,
        agent_id=agent_id,
        enable_memory_retrieval=False,
        generation_options={"temperature": 0, "top_p": 1},
    ):
        events.append(event)
    return events


async def test_live_sub_agent_call_stream_contract(e2e_system, e2e_user):
    """真实 LLM 下验证 CALL 与子代理流式事件契约。"""
    prompt = (
        "请严格按下面流程执行：\n"
        "1) 通过 CALL 调用子代理 coder_doll。\n"
        "2) 给 coder_doll 的任务：写一段 Python 排序代码（例如快速排序或归并排序），并返回代码。\n"
        "3) 你收到子代理返回后，再给出简短总结。"
        # "要求：第一步必须出现 CALL，且目标 alias 必须是 coder_doll。"
    )

    last_events: list[dict[str, Any]] = []
    for _ in range(2):
        user_id = await _caller_with_coder_doll(e2e_system, e2e_user)
        last_events = await _collect_stream_events(e2e_system, prompt, user_id=user_id)
        done = next((e for e in last_events if e.get("event") == "done"), None)
        if done and "CALL" in _commands(last_events):
            break

    done = next((e for e in last_events if e.get("event") == "done"), None)
    assert done is not None, f"未收到 done 事件。events={_event_summary(last_events)}"
    assert done["data"].get(
        "final_text"
    ), f"done.final_text 为空。events={_event_summary(last_events)}"
    assert "CALL" in _commands(last_events), f"未触发 CALL。events={_event_summary(last_events)}"

    sub_start = next((e for e in last_events if e["event"] == "sub_agent_start"), None)
    assert sub_start is not None, f"缺少 sub_agent_start。events={_event_summary(last_events)}"
    assert (
        sub_start.get("data", {}).get("agent_id") == "coder_doll"
    ), f"CALL 目标不是 coder_doll。events={_event_summary(last_events)}"
    assert any(
        e["event"] == "sub_agent_end" for e in last_events
    ), f"缺少 sub_agent_end。events={_event_summary(last_events)}"
    assert any(
        e["event"] in {"token", "mtp_start", "mtp_result"}
        and e.get("data", {}).get("scope") == "sub"
        for e in last_events
    ), f"缺少 scope=sub 的子代理流事件。events={_event_summary(last_events)}"


async def test_live_sub_agent_disallow_nested_call(e2e_system, e2e_user):
    """真实 LLM 下验证子代理侧不应出现 CALL（星型拓扑约束）。"""
    prompt = (
        "请通过 CALL 委派 coder_doll 完成任务：写一段 Python 排序代码并返回。"
        "同时要求子代理尝试进一步分解任务。"
        "你必须通过 CALL 发起委派，最后由你汇总。"
    )

    user_id = await _caller_with_coder_doll(e2e_system, e2e_user)
    events = await _collect_stream_events(e2e_system, prompt, user_id=user_id)
    done = next((e for e in events if e.get("event") == "done"), None)
    assert done is not None, f"未收到 done 事件。events={_event_summary(events)}"

    # 子作用域下若出现 mtp_start，动词不应为 CALL（depth>=1 禁止 CALL）
    sub_call_events = [
        e
        for e in events
        if e.get("event") == "mtp_start"
        and e.get("data", {}).get("scope") == "sub"
        and str(e.get("data", {}).get("verb", "")).upper() == "CALL"
    ]
    assert (
        not sub_call_events
    ), f"检测到子代理 CALL 事件，违反星型拓扑。events={_event_summary(events)}"
