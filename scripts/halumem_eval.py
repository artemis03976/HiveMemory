"""在 HaluMem 上评测记忆生成：驱动一个试验实例逐次会话写入记忆，按官方协议判分并汇总。

HaluMem 先由记忆要点生成对话，要点就是标准答案。官方协议逐次会话进行：导入一次会话的对话，
取出系统在这次会话里写出的记忆，再对每条更新要点以其内容检索前 10 条记忆；判分有三类，
每次调用一个评判模型：

- 完整度：每条不走更新评测的要点一次，看它在这次写出的记忆里有没有；干扰要点也判，用来
  计算“不记干扰”；
- 准确度：每条写出的记忆一次，对照本次对话与非干扰要点，看有没有编造；
- 更新：每条有检索结果的更新要点一次，看检索到的记忆是否改对。

问答与依赖日期的题不在这里；对话按轮导入，不带原始时间。

子命令：

- ``prompts``：从官方评测代码 ``eval/eval_tools.py`` 中取出三份评判提示词的原文，写成文本文件；
  只解析源码，不执行。数据与提示词的许可是 CC BY-NC-ND，只在本机使用，不进仓库；
- ``run``：建立（或续用）一个试验实例，逐个用户、按顺序逐次会话导入、结算、取出写出的记忆，
  并为更新要点检索；结果写到 ``<run>/driven/<用户>.jsonl``，一次会话一行。导入按外部标识
  去重，中断后再运行会从最后一个完成的会话之后继续；
- ``judge``：按官方协议判分，结果写到 ``<run>/judged.jsonl``，可以续跑；评判模型不能与这次
  运行的生成模型相同；随机抽一部分调用再判一次，记下两次是否一致；
- ``report``：按官方的口径汇总指标，在运行目录写出一份 Markdown 报告与一份 JSON 指标；可以把
  几次运行并排，并向结果账本追加一条。

用户在数据集中按行序编号，试验实例中的用户 id 为 ``halumem-<编号>``，每个用户有自己的
Workspace。生成模型与结算方式写在试验实例的配置里，评判模型按开发 checkout 的
``configs/models.yaml`` 解析，凭证来自 ``configs/providers.secrets.yaml`` 与 ``.env``。

典型用法（试跑：1 个用户的前 10 次会话）::

    python scripts/halumem_eval.py prompts path/to/eval_tools.py .hivememory/lab/halumem/prompts
    python scripts/halumem_eval.py run .hivememory/lab/datasets/halumem/HaluMem-Medium.jsonl \\
        .hivememory/lab/runs/pilot-one-shot --users 15 --sessions 10 --settle-mode one_shot
    python scripts/halumem_eval.py judge .hivememory/lab/datasets/halumem/HaluMem-Medium.jsonl \\
        .hivememory/lab/runs/pilot-one-shot --prompts .hivememory/lab/halumem/prompts \\
        --judge-model gpt-6-luna
    python scripts/halumem_eval.py report .hivememory/lab/runs/pilot-one-shot
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from memory_replay import gather_limited, read_jsonl
from trial_instance import (
    InstanceEndpoints,
    TrialInstanceError,
    add_spec_arguments,
    create,
    load_spec,
    run_instance,
    spec_from_args,
)

from hivememory.config.app import provider_credentials_from_env
from hivememory.core.errors.model import ModelProviderError
from hivememory.core.models.model_definition import ModelSelection
from hivememory.infrastructure.llm.litellm_service import LiteLLMService
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

SOURCE = "halumem"

# 官方评测代码中三份评判提示词的变量名，与本脚本使用的文件名。
PROMPT_NAMES = {
    "integrity": "EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY",
    "accuracy": "EVALUATION_PROMPT_FOR_MEMORY_ACCURACY",
    "update": "EVALUATION_PROMPT_FOR_UPDATE_MEMORY",
}

# 连接器为路由交出的之前几轮，与 harness 插件相同。
RECENT_TURNS = 3
# 官方协议以更新要点的内容检索前 10 条记忆。
UPDATE_SEARCH_LIMIT = 10
MEMORY_PAGE_SIZE = 100

IMPORT_ATTEMPTS = 4
SETTLE_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 2
UNFINISHED_TASK_STATUSES = frozenset({"pending", "running"})

# 抽样再判一次的比例，用来估计评判的稳定性。
RECHECK_RATIO = 0.1


def user_id(index: int) -> str:
    return f"halumem-{index:02d}"


def load_users(dataset: Path) -> list[dict[str, Any]]:
    return read_jsonl(dataset)


def memory_text(memory: dict[str, Any]) -> str:
    """一条记忆交给评判的文本：标题、摘要与正文。"""
    return f"{memory['title']}\n{memory['summary']}\n{memory['content']}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


# ---------------------------------------------------------------- prompts


def extract_prompts(eval_tools: Path, out: Path) -> None:
    """按变量名取出字符串常量；只解析源码，不执行它。"""
    tree = ast.parse(eval_tools.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    found[target.id] = node.value.value
    out.mkdir(parents=True, exist_ok=True)
    for kind, name in PROMPT_NAMES.items():
        if name not in found:
            raise SystemExit(f"{eval_tools} 中没有字符串常量 {name}")
        (out / f"{kind}.txt").write_text(found[name], encoding="utf-8")
    print(f"wrote {', '.join(PROMPT_NAMES)} to {out}")


# ---------------------------------------------------------------- run


class DriverError(RuntimeError):
    """一次会话无法继续；这个用户停在这里，其余用户照常。"""


@dataclass(frozen=True)
class Client:
    """试验实例的 HTTP 接口：导入以导入来源认证，其余以管理凭据、按用户的 Workspace。"""

    http: httpx.AsyncClient
    endpoints: InstanceEndpoints

    def management(self, user: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.endpoints.management_token}", "X-User-ID": user}

    async def import_turn(self, user: str, body: dict[str, Any]) -> dict[str, Any]:
        """导入一轮；连接失败与 5xx 退避重试，422 与 413 作为拒绝返回，其余 4xx 停下。"""
        headers = {
            "Authorization": f"Bearer {self.endpoints.import_token}",
            "X-Principal-ID": self.endpoints.principal_id,
            "X-User-ID": user,
        }
        for attempt in range(IMPORT_ATTEMPTS):
            try:
                response = await self.http.post("/api/v1/import/turns", json=body, headers=headers)
            except httpx.TransportError as error:
                if attempt == IMPORT_ATTEMPTS - 1:
                    raise DriverError(f"导入连接失败: {type(error).__name__}") from error
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (413, 422):
                    return {"status": "rejected", "detail": response.text[:500]}
                if response.status_code < 500:
                    raise DriverError(f"导入被拒绝 {response.status_code}: {response.text[:500]}")
                if attempt == IMPORT_ATTEMPTS - 1:
                    raise DriverError(f"导入失败 {response.status_code}: {response.text[:500]}")
            await asyncio.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    async def get(self, user: str, path: str, **params: Any) -> Any:
        response = await self.http.get(path, params=params, headers=self.management(user))
        if response.status_code != 200:
            raise DriverError(f"GET {path} {response.status_code}: {response.text[:500]}")
        return response.json()

    async def settle(self, user: str, topic_id: str) -> None:
        response = await self.http.post(
            f"/api/v1/topics/{topic_id}/settle", headers=self.management(user)
        )
        # 话题在结算之前被 LRU 或空闲结算带走时返回 404，它的任务照样在任务列表里。
        if response.status_code not in (200, 404):
            raise DriverError(f"结算失败 {response.status_code}: {response.text[:500]}")

    async def memories(self, user: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": MEMORY_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            page = await self.get(user, "/api/v1/memories", **params)
            for memory in page["memories"]:
                result[memory["id"]] = memory
            cursor = page.get("next_cursor")
            if not cursor:
                return result

    async def search(self, user: str, query: str) -> list[dict[str, Any]]:
        page = await self.get(user, "/api/v1/memories", query=query, limit=UPDATE_SEARCH_LIMIT)
        return page["memories"]


async def settle_and_drain(client: Client, user: str) -> dict[str, int]:
    """结算这个 Workspace 的全部活跃话题，等到生成任务（含之后的合并任务）都结束。"""
    topics = await client.get(user, "/api/v1/topics")
    for topic in topics["topics"]:
        await client.settle(user, topic["topic_id"])
    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    while True:
        tasks = (await client.get(user, "/api/v1/memory-tasks"))["tasks"]
        unfinished = [t for t in tasks if t["status"] in UNFINISHED_TASK_STATUSES]
        if not unfinished:
            return dict(collections.Counter(t["status"] for t in tasks))
        if time.monotonic() > deadline:
            raise DriverError(
                f"{len(unfinished)} 个生成任务在 {SETTLE_TIMEOUT_SECONDS} 秒内没有结束"
            )
        await asyncio.sleep(POLL_SECONDS)


def session_turns(session: dict[str, Any]) -> list[tuple[int, str, str]]:
    """把对话按轮配成 (轮序号, 用户消息, 助手回复)；角色不成对时拒绝这次会话。"""
    dialogue = session["dialogue"]
    if len(dialogue) % 2:
        raise DriverError("对话的消息数不是偶数")
    turns = []
    for user_message, reply in zip(dialogue[::2], dialogue[1::2], strict=True):
        if user_message["role"] != "user" or reply["role"] != "assistant":
            raise DriverError("对话不是用户与助手交替")
        turns.append(
            (int(user_message["dialogue_turn"]), user_message["content"], reply["content"])
        )
    return turns


async def import_session(
    client: Client, user: str, session_index: int, session: dict[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    """按轮导入一次会话；返回成功导入的轮数与被拒绝的轮。"""
    conversation = f"{user}-s{session_index:03d}"
    turns = session_turns(session)
    imported = 0
    rejected = []
    for position, (turn, user_message, reply) in enumerate(turns):
        recent = [
            {"user_message": u, "assistant_text": a, "external_turn_id": str(t)}
            for t, u, a in turns[max(0, position - RECENT_TURNS) : position]
        ]
        body = {
            "source": SOURCE,
            "external_conversation_id": conversation,
            "external_turn_id": str(turn),
            "user_message": user_message,
            "recent_turns": recent,
            "outcome": "completed",
            "events": [{"kind": "assistant_message", "sequence": 1, "content": reply}],
            "agent_id": client.endpoints.agent_id,
        }
        result = await client.import_turn(user, body)
        if result["status"] == "rejected":
            rejected.append({"turn": turn, "detail": result["detail"]})
        else:
            imported += 1
    return imported, rejected


def changed_memories(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """这次会话写出的记忆：新建的，以及内容被改写过（``updated_at`` 变了）的。"""
    changed = []
    for memory_id, memory in after.items():
        previous = before.get(memory_id)
        if previous is None:
            change = "created"
        elif previous["updated_at"] != memory["updated_at"]:
            change = "updated"
        else:
            continue
        changed.append(
            {
                "id": memory_id,
                "change": change,
                "memory_type": memory["memory_type"],
                "title": memory["title"],
                "summary": memory["summary"],
                "content": memory["content"],
            }
        )
    return changed


async def drive_session(
    client: Client, user: str, session_index: int, session: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    await settle_and_drain(client, user)
    before = await client.memories(user)
    imported, rejected = await import_session(client, user, session_index, session)
    imported_at = time.monotonic()
    task_statuses = await settle_and_drain(client, user)
    settled_at = time.monotonic()
    after = await client.memories(user)
    update_searches: dict[str, list[str]] = {}
    for index, point in enumerate(session["memory_points"]):
        if point["is_update"] == "True" and point["original_memories"]:
            found = await client.search(user, point["memory_content"])
            update_searches[str(index)] = [memory_text(memory) for memory in found]
    return {
        "user": user,
        "session_index": session_index,
        "imported_turns": imported,
        "rejected_turns": rejected,
        "extracted_memories": changed_memories(before, after),
        "removed_memory_ids": sorted(set(before) - set(after)),
        "memory_count_after": len(after),
        "task_statuses": task_statuses,
        "update_searches": update_searches,
        "seconds": {
            "import": round(imported_at - started, 1),
            "settle": round(settled_at - imported_at, 1),
            "search": round(time.monotonic() - settled_at, 1),
        },
    }


async def drive_user(
    client: Client, run_dir: Path, index: int, data: dict[str, Any], session_limit: int | None
) -> None:
    user = user_id(index)
    path = run_dir / "driven" / f"{user}.jsonl"
    done = {row["session_index"] for row in read_jsonl_if_exists(path)}
    sessions = data["sessions"][:session_limit] if session_limit else data["sessions"]
    for session_index, session in enumerate(sessions):
        if session_index in done or session.get("is_generated_qa_session"):
            continue
        try:
            row = await drive_session(client, user, session_index, session)
        except DriverError as error:
            append_jsonl(
                run_dir / "driven" / f"{user}.errors.jsonl",
                {"session_index": session_index, "error": str(error), "at": _now()},
            )
            print(f"{user} s{session_index:03d} failed: {error}", flush=True)
            return
        append_jsonl(path, row)
        print(
            f"{user} s{session_index:03d} turns={row['imported_turns']} "
            f"written={len(row['extracted_memories'])} library={row['memory_count_after']} "
            f"import={row['seconds']['import']}s settle={row['seconds']['settle']}s",
            flush=True,
        )


async def drive(args: argparse.Namespace) -> None:
    users = load_users(args.dataset)
    indexes = [int(part) for part in args.users.split(",")]
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        create(run_dir, spec_from_args(args, [user_id(i) for i in indexes]))
        manifest = {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "users": indexes,
            "sessions": args.sessions,
        }
        (run_dir / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    spec = load_spec(run_dir)
    missing = [user_id(i) for i in indexes if user_id(i) not in spec.users]
    if missing:
        raise SystemExit(f"试验实例没有登记这些用户：{', '.join(missing)}")
    with run_instance(run_dir) as endpoints:
        async with httpx.AsyncClient(
            base_url=endpoints.base_url, trust_env=False, timeout=600
        ) as http:
            client = Client(http=http, endpoints=endpoints)
            jobs = [
                lambda i=i: drive_user(client, run_dir, i, users[i], args.sessions) for i in indexes
            ]
            await gather_limited(jobs, args.concurrency)


# ---------------------------------------------------------------- judge


def dialogue_text(session: dict[str, Any]) -> str:
    """官方评分器交给准确度评判的对话格式：每条消息带时间，助手回复后空一行。"""
    lines = []
    for message in session["dialogue"]:
        lines.append(f"[{message['timestamp']}]{message['role']}: {message['content']}")
        if message["role"] == "assistant":
            lines.append("")
    return "\n".join(lines)


def judge_calls(users: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    """按官方评分器的规则列出这次运行的全部评判调用。"""
    calls: list[dict[str, Any]] = []
    for path in sorted((run_dir / "driven").glob("halumem-*.jsonl")):
        if path.name.endswith(".errors.jsonl"):
            continue
        for row in read_jsonl(path):
            index = int(row["user"].removeprefix("halumem-"))
            session = users[index]["sessions"][row["session_index"]]
            prefix = f"{row['user']}/{row['session_index']}"
            extracted = row["extracted_memories"]
            extracted_text = "\n\n".join(memory_text(memory) for memory in extracted)
            for i, point in enumerate(session["memory_points"]):
                base = {
                    "user": row["user"],
                    "session_index": row["session_index"],
                    "point_index": i,
                    "memory_source": point["memory_source"],
                    "importance": float(point["importance"]),
                }
                searched = row["update_searches"].get(str(i))
                if point["is_update"] == "True" and searched:
                    calls.append(
                        {
                            **base,
                            "key": f"{prefix}/update/{i}",
                            "kind": "update",
                            "fields": {
                                "memories": "\n".join(searched),
                                "updated_memory": point["memory_content"],
                                "original_memory": "\n".join(point["original_memories"]),
                            },
                        }
                    )
                else:
                    calls.append(
                        {
                            **base,
                            "key": f"{prefix}/integrity/{i}",
                            "kind": "integrity",
                            "fields": {
                                "memories": extracted_text,
                                "expected_memory_point": point["memory_content"],
                            },
                        }
                    )
            golden = "\n".join(
                p["memory_content"]
                for p in session["memory_points"]
                if p["memory_source"] != "interference"
            )
            dialogue = dialogue_text(session)
            for memory in extracted:
                calls.append(
                    {
                        "key": f"{prefix}/accuracy/{memory['id']}",
                        "kind": "accuracy",
                        "user": row["user"],
                        "session_index": row["session_index"],
                        "memory_id": memory["id"],
                        "fields": {
                            "dialogue": dialogue,
                            "golden_memories": golden,
                            "candidate_memory": memory_text(memory),
                        },
                    }
                )
    return calls


def is_recheck(key: str) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return digest[0] < 256 * RECHECK_RATIO


def judge_result(kind: str, payload: Any) -> dict[str, Any]:
    """按官方格式取出判分；字段缺失或取值不合法时为 None，单独计数。"""
    if not isinstance(payload, dict):
        return {"score": None}
    if kind == "integrity":
        score = payload.get("score")
        return {"score": int(score) if str(score) in {"0", "1", "2"} else None}
    if kind == "accuracy":
        score = payload.get("accuracy_score")
        included = str(payload.get("is_included_in_golden_memories", "false")).lower()
        return {
            "score": int(score) if str(score) in {"0", "1", "2"} else None,
            "included": included == "true",
        }
    result = payload.get("evaluation_result")
    valid = {"Correct", "Hallucination", "Omission", "Other"}
    return {"score": result if result in valid else None}


async def judge(args: argparse.Namespace) -> None:
    spec = load_spec(args.run_dir)
    if args.judge_model == spec.generation_model:
        raise SystemExit("评判模型与这次运行的生成模型相同；换一个评判模型")
    prompts = {kind: (args.prompts / f"{kind}.txt").read_text("utf-8") for kind in PROMPT_NAMES}
    registry = ModelRegistry(
        provider_registry=ProviderRegistry(env_providers=provider_credentials_from_env())
    )
    params = registry.resolve(ModelSelection(model_id=args.judge_model, temperature=0.0))
    params = params.model_copy(update={"json_output": True})
    llm = LiteLLMService()
    out = args.run_dir / "judged.jsonl"
    done = {(row["key"], row["attempt"]) for row in read_jsonl_if_exists(out)}
    calls = judge_calls(load_users(args.dataset), args.run_dir)
    pending = [
        (call, attempt)
        for call in calls
        for attempt in ((1, 2) if is_recheck(call["key"]) else (1,))
        if (call["key"], attempt) not in done
    ]
    print(f"calls={len(calls)} pending={len(pending)} judge_model={args.judge_model}", flush=True)

    async def run(call: dict[str, Any], attempt: int) -> None:
        prompt = prompts[call["kind"]].format(**call["fields"])
        tokens = None
        try:
            completion = await llm.complete([{"role": "user", "content": prompt}], params)
            tokens = completion.usage.total_tokens if completion.usage else None
            payload: Any = parse_llm_json(completion.text)
        except (ModelProviderError, JSONParseError) as error:
            payload = {"error": type(error).__name__}
        row = {key: value for key, value in call.items() if key != "fields"}
        append_jsonl(
            out,
            {
                **row,
                "attempt": attempt,
                **judge_result(call["kind"], payload),
                "tokens": tokens,
                "raw": payload,
                "judge_model": args.judge_model,
            },
        )

    jobs = [lambda c=c, a=a: run(c, a) for c, a in pending]
    await gather_limited(jobs, args.concurrency)


# ---------------------------------------------------------------- report


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def metrics(run_dir: Path) -> dict[str, Any]:
    """官方评分器的口径；记录只取第一次判分，第二次只用来计算一致率。"""
    rows = read_jsonl_if_exists(run_dir / "judged.jsonl")
    first = {row["key"]: row for row in rows if row["attempt"] == 1}
    second = {row["key"]: row for row in rows if row["attempt"] == 2}
    integrity = [r for r in first.values() if r["kind"] == "integrity"]
    target = [r for r in integrity if r["memory_source"] != "interference"]
    interference = [r for r in integrity if r["memory_source"] == "interference"]
    accuracy = [r for r in first.values() if r["kind"] == "accuracy"]
    update = [r for r in first.values() if r["kind"] == "update"]

    recall = ratio(sum(1 for r in target if r["score"] == 2), len(target))
    weighted_recall = ratio(
        sum(0.5 * r["score"] * r["importance"] for r in target if r["score"] is not None),
        sum(r["importance"] for r in target),
    )
    included = [r for r in accuracy if r.get("included")]
    target_accuracy = ratio(
        sum(0.5 * r["score"] for r in included if r["score"] is not None), len(included)
    )
    weighted_accuracy = ratio(
        sum(0.5 * r["score"] for r in accuracy if r["score"] is not None), len(accuracy)
    )
    f1 = (
        2 * target_accuracy * recall / (target_accuracy + recall)
        if target_accuracy and recall
        else None
    )
    update_counts = collections.Counter(r["score"] for r in update)
    rechecked = [(first[k], second[k]) for k in second if k in first]
    agreement = ratio(sum(1 for a, b in rechecked if a["score"] == b["score"]), len(rechecked))

    driven = [
        row
        for path in sorted((run_dir / "driven").glob("halumem-*.jsonl"))
        if not path.name.endswith(".errors.jsonl")
        for row in read_jsonl(path)
    ]
    written = [len(row["extracted_memories"]) for row in driven]
    lengths = [len(m["content"]) for row in driven for m in row["extracted_memories"]]
    return {
        "sessions": len(driven),
        "turns": sum(row["imported_turns"] for row in driven),
        "rejected_turns": sum(len(row["rejected_turns"]) for row in driven),
        "memories_written": sum(written),
        "memories_per_session": ratio(sum(written), len(written)),
        "mean_memory_chars": ratio(sum(lengths), len(lengths)),
        "integrity": {
            "recall": recall,
            "weighted_recall": weighted_recall,
            "points": len(target),
            "invalid": sum(1 for r in target if r["score"] is None),
        },
        "interference_resistance": ratio(
            sum(1 for r in interference if r["score"] == 0), len(interference)
        ),
        "accuracy": {
            "target_accuracy": target_accuracy,
            "weighted_accuracy": weighted_accuracy,
            "memories": len(accuracy),
            "included_in_golden": len(included),
            "invalid": sum(1 for r in accuracy if r["score"] is None),
        },
        "extraction_f1": f1,
        "update": {
            "points": len(update),
            **{
                f"{name.lower()}_ratio": ratio(update_counts.get(name, 0), len(update))
                for name in ("Correct", "Hallucination", "Omission", "Other")
            },
            "invalid": update_counts.get(None, 0),
        },
        "judge": {
            "calls": len(first),
            "invalid": sum(1 for r in first.values() if r["score"] is None),
            "recheck_agreement": agreement,
            "rechecked": len(rechecked),
            "tokens": sum(r["tokens"] or 0 for r in rows),
        },
        "seconds": {
            phase: round(sum(row["seconds"][phase] for row in driven), 1)
            for phase in ("import", "settle", "search")
        },
        "instance_usage": instance_usage(run_dir),
    }


LIBRARIAN_LOG = re.compile(r"馆长结算结束: .*steps=(\d+), .*total_tokens=(\d+)")


def instance_usage(run_dir: Path) -> dict[str, Any]:
    """试验实例的用量：路由与生成任务的次数取自运行时事件，馆长的 token 取自每次结算的日志。

    其余模型调用的 token 没有记录；精确的花费以提供商的账单为准。
    """
    events: collections.Counter[str] = collections.Counter()
    logs = run_dir / "instance" / "logs"
    for path in sorted(logs.glob("runtime-events.jsonl*")):
        for row in read_jsonl(path):
            events[row.get("event_type", "")] += 1
    librarian_steps = librarian_tokens = librarian_settlements = 0
    for log in sorted(logs.glob("backend.log*")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if match := LIBRARIAN_LOG.search(line):
                librarian_settlements += 1
                librarian_steps += int(match.group(1))
                librarian_tokens += int(match.group(2))
    return {
        "routing_calls": events["gateway.workflow.completed"],
        "generation_tasks": events["memory.task.completed"] + events["memory.task.failed"],
        "failed_generation_tasks": events["memory.task.failed"],
        "librarian_settlements": librarian_settlements,
        "librarian_steps": librarian_steps,
        "librarian_tokens": librarian_tokens,
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def report_table(runs: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> str:
    header = "| 指标 | " + " | ".join(name for name, _, _ in runs) + " |"
    lines = [header, "|:---|" + "---:|" * len(runs)]

    def row(label: str, values: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("结算方式", [spec["settle_mode"] for _, spec, _ in runs])
    row("生成模型", [spec["generation_model"] for _, spec, _ in runs])
    row("会话 / 轮", [f"{m['sessions']} / {m['turns']}" for _, _, m in runs])
    row("完整度（召回）", [percent(m["integrity"]["recall"]) for _, _, m in runs])
    row("完整度（按重要度加权）", [percent(m["integrity"]["weighted_recall"]) for _, _, m in runs])
    row(
        "准确度（落在要点内的记忆）",
        [percent(m["accuracy"]["target_accuracy"]) for _, _, m in runs],
    )
    row("准确度（全部记忆）", [percent(m["accuracy"]["weighted_accuracy"]) for _, _, m in runs])
    row("抽取 F1", [percent(m["extraction_f1"]) for _, _, m in runs])
    row("不记干扰", [percent(m["interference_resistance"]) for _, _, m in runs])
    row("更新：正确", [percent(m["update"]["correct_ratio"]) for _, _, m in runs])
    row("更新：幻觉", [percent(m["update"]["hallucination_ratio"]) for _, _, m in runs])
    row("更新：遗漏", [percent(m["update"]["omission_ratio"]) for _, _, m in runs])
    row("每次会话写出的记忆", [number(m["memories_per_session"]) for _, _, m in runs])
    row("记忆正文平均字数", [number(m["mean_memory_chars"]) for _, _, m in runs])
    row("评判无效 / 调用", [f"{m['judge']['invalid']} / {m['judge']['calls']}" for _, _, m in runs])
    row("评判一致率", [percent(m["judge"]["recheck_agreement"]) for _, _, m in runs])
    row(
        "耗时（导入 / 结算）",
        [f"{m['seconds']['import']}s / {m['seconds']['settle']}s" for _, _, m in runs],
    )
    row("话题路由调用", [str(m["instance_usage"]["routing_calls"]) for _, _, m in runs])
    row(
        "生成任务（失败）",
        [
            f"{m['instance_usage']['generation_tasks']} ({m['instance_usage']['failed_generation_tasks']})"
            for _, _, m in runs
        ],
    )
    row(
        "馆长步数 / token",
        [
            f"{m['instance_usage']['librarian_steps']} / {m['instance_usage']['librarian_tokens']}"
            for _, _, m in runs
        ],
    )
    row("评判 token", [str(m["judge"]["tokens"]) for _, _, m in runs])
    return "\n".join(lines)


def report(args: argparse.Namespace) -> None:
    runs = []
    for run_dir in [args.run_dir, *args.compare]:
        spec = json.loads((run_dir / "instance.json").read_text(encoding="utf-8"))
        computed = metrics(run_dir)
        (run_dir / "metrics.json").write_text(
            json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        runs.append((run_dir.name, spec["spec"], computed))
    record = json.loads((args.run_dir / "instance.json").read_text(encoding="utf-8"))
    dataset = json.loads((args.run_dir / "dataset.json").read_text(encoding="utf-8"))
    text = "\n".join(
        [
            f"# HaluMem 评测：{args.run_dir.name}",
            "",
            f"- 生成于 {_now()}；仓库提交 `{record['repo_commit']}`",
            f"- 数据集 `{dataset['dataset']}`（sha256 `{dataset['dataset_sha256'][:12]}`），"
            f"用户 {dataset['users']}，每人会话上限 {dataset['sessions'] or '全部'}",
            "",
            report_table(runs),
            "",
        ]
    )
    (args.run_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    if args.ledger:
        entry = "\n".join(
            [
                f"## {_now()[:10]} · 试验 · HaluMem：{' vs '.join(name for name, _, _ in runs)}",
                "",
                report_table(runs),
                "",
                f"- 报告：`{args.run_dir / 'report.md'}`",
                "",
            ]
        )
        with args.ledger.open("a", encoding="utf-8") as file:
            file.write(entry)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 HaluMem 上评测记忆生成")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prompts", help="从官方 eval_tools.py 取出评判提示词")
    p.add_argument("eval_tools", type=Path)
    p.add_argument("out", type=Path)

    p = commands.add_parser("run", help="建立或续用试验实例，逐次会话写入记忆")
    p.add_argument("dataset", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--users", required=True, help="逗号分隔的用户行序号")
    p.add_argument("--sessions", type=int, default=None, help="每个用户最多处理的会话数")
    p.add_argument("--concurrency", type=int, default=5, help="同时处理的用户数")
    add_spec_arguments(p)

    p = commands.add_parser("judge", help="按官方协议判分")
    p.add_argument("dataset", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prompts", type=Path, required=True, help="prompts 子命令写出的目录")
    p.add_argument("--judge-model", required=True, help="models.yaml 中的模型 id")
    p.add_argument("--concurrency", type=int, default=10)

    p = commands.add_parser("report", help="汇总指标，写出报告")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--compare", type=Path, nargs="*", default=[], help="并排的其他运行目录")
    p.add_argument("--ledger", type=Path, default=None, help="追加结果的账本文件")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prompts":
            extract_prompts(args.eval_tools, args.out)
        elif args.command == "run":
            asyncio.run(drive(args))
        elif args.command == "judge":
            asyncio.run(judge(args))
        elif args.command == "report":
            report(args)
    except TrialInstanceError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
