"""在已有数据上离线回放记忆的生成与分类，比较同一份材料多次运行的结果。

回放只读已有数据，结果写到调用方给出的位置；不启动后端，不写任何记忆。模型按主配置的
``patchouli.llm`` 解析，被动提取器按 ``patchouli.generation.extractor`` 组装，与日常的
记忆整理相同；凭证与后端一样来自环境变量和 ``.env``。

子命令：

- ``export``：从一个 Workspace 的 Artifact 目录导出被动结算的生成材料，以及每条记忆建成时的
  类型与正文；Artifact 根目录下还留有迁移前的旧副本，所以不从根目录导出；
- ``extract``：把生成材料交给现在的被动提取器，或交给一份自定义的系统提示词，每份重复几次；
- ``classify``：把记忆按批交给一份分类提示词，每条重复几次；
- ``agreement``：统计 ``classify`` 结果中某个字段多次运行是否一致，以及两份结果的多数判定是否相同；
- ``turns-claude-code``：用 Claude Code 连接器的读取器从头重读本机的会话记录，导出带响应序号的
  轮次，不经过服务端；
- ``turns-snapshots``：从一个 Workspace 的 Artifact 目录导出交互快照中的轮次，已有数据没有响应
  序号；
- ``segments``：按不同的每段响应上限切分导出的轮次，统计每轮段数与每段响应数；
- ``extract-turns``：把导出的轮次按会话拼成生成材料，交给现在的被动提取提示词，每份重复几次，
  记下模型给出的段引用；
- ``citations``：统计 ``extract-turns`` 结果中引用的合格率与多次运行的一致性，并抽出引用供人工核对。

自定义提示词是一个文本文件，原样作为系统消息：``extract`` 的用户消息是渲染好的生成材料；
``classify`` 的用户消息是一批记忆，提示词要让模型输出
``{"results": [{"id": "<原样照抄的 id>", ...}]}``，``agreement`` 按其中的字段统计。

轮次文件每行一轮：``id``、``conversation_id``、``turn_id``、``outcome`` 与以序号 0 的用户
消息开头的 ``events``（``TurnEvent`` 的 JSON）。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import re
import statistics
import sys
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hivememory.config.app import load_app_config, provider_credentials_from_env
from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors.model import ModelProviderError
from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    TurnEvent,
    TurnRecord,
    validate_turn_events,
)
from hivememory.engines.generation.extractor import LLMMemoryExtractor, create_extractor
from hivememory.engines.generation.models import (
    ExtractionResult,
    GenerationContext,
    segment_label,
)
from hivememory.engines.perception.segments import segment_turn
from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams
from hivememory.infrastructure.llm.litellm_service import LiteLLMService
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.system.services.import_bus import ImportedTurn
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

# 一批记忆中每条正文最多给模型看的字符数；分类只需要判断它是什么，不需要全文。
CLASSIFY_CONTENT_CHARS = 1500

# 连接器不在 hivememory 包内，按插件运行时的方式从仓库目录导入。
CONNECTOR_ROOT = Path(__file__).resolve().parents[1] / "integrations" / "hivememory"

# 回放材料中各轮的来源身份；只影响来源记录，不影响材料与引用。
REPLAY_IDENTITY = ActorIdentity(user_id="replay", agent_id="replay")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- export


def upgrade_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """把较早结算记录中的轮次改成现在的形状，使生成材料能按现在的模型解码。

    较早的记录把一轮拆成 ``user_query``、``assistant_final_text`` 与 ``action_entries``，
    现在的材料只有 ``user_message`` 与 ``material_entries``；回复正文按材料条目的写法
    追加在末尾。分段之前的材料没有轮次标识与段，补成空串与空列表，渲染时不印段标签。
    """
    if "user_message" not in turn:
        entries = [e if e.startswith("- ") else f"- {e}" for e in turn.get("action_entries") or []]
        if turn.get("assistant_final_text"):
            entries.append(f"[Assistant]: {turn['assistant_final_text']}")
        turn = {
            "outcome": turn.get("outcome", "completed"),
            "user_message": turn.get("user_query", ""),
            "material_entries": entries,
            "identity": turn["identity"],
        }
    return {"turn_id": "", "segments": [], **turn}


def export(artifacts: Path, out: Path) -> None:
    """导出 ``memories.jsonl`` 与 ``settlements.jsonl``。

    记录按 JSON 原样读取，不经生产模型解码：较早的记录属于旧 schema，生产模型只解码
    现在的 schema。一条记忆建成时的类型、标题、摘要与正文取自它的第 1 版快照。
    """
    creations: list[dict[str, Any]] = []
    versions: dict[str, dict[str, Any]] = {}
    for path in artifacts.rglob("*.json"):
        if path.name.startswith("."):
            continue  # 存储自己的索引文件
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("artifact_type") == "memory_creation":
            creations.append(record)
        elif record.get("artifact_type") == "memory_version":
            versions[record["artifact_id"]] = record

    memories = []
    settlements: dict[str, dict[str, Any]] = {}
    for creation in sorted(creations, key=lambda c: c["created_at"]):
        settlement = next(
            (
                ref["artifact_id"]
                for ref in creation.get("source_artifacts", [])
                if ref.get("artifact_type") == "interaction"
            ),
            None,
        )
        version_ref = creation.get("initial_version_ref") or {}
        first_version = versions.get(version_ref.get("artifact_id", ""), {})
        snapshot = first_version.get("snapshot_after") or {}
        index = snapshot.get("index") or {}
        agents = (creation.get("provenance") or {}).get("contributing_agent_ids", [])
        memories.append(
            {
                "memory_id": creation["memory_id"],
                "source_intent": creation["source_intent"],
                "settlement": settlement,
                "created_at": creation["created_at"],
                "agents": agents,
                "memory_type": index.get("memory_type"),
                "title": index.get("title", ""),
                "summary": index.get("summary", ""),
                "content": (snapshot.get("payload") or {}).get("content", ""),
            }
        )
        # 被动结算一次产出的几条记忆共用同一份生成材料，只导出一次。
        if creation["source_intent"] == "SYSTEM" and settlement and settlement not in settlements:
            view = creation["generation_view"]
            view = {**view, "turns": [upgrade_turn(t) for t in view.get("turns", [])]}
            try:
                GenerationContext.model_validate(view)  # 导出时就暴露解码不了的材料
            except ValidationError as error:
                raise SystemExit(f"结算 {settlement} 的生成材料无法解码：{error}") from error
            settlements[settlement] = {
                "id": settlement,
                "created_at": creation["created_at"],
                "agents": agents,
                "generation_view": view,
            }

    created = collections.Counter(
        m["settlement"] for m in memories if m["source_intent"] == "SYSTEM"
    )
    for settlement in settlements.values():
        settlement["created_count"] = created[settlement["id"]]
    write_jsonl(out / "memories.jsonl", memories)
    write_jsonl(out / "settlements.jsonl", settlements.values())
    intents = collections.Counter(m["source_intent"] for m in memories)
    print(f"memories={len(memories)} by_intent={dict(intents)} settlements={len(settlements)}")


# ---------------------------------------------------------------- model


def librarian_model() -> tuple[BaseLLMService, LLMRequestParams, ExtractorConfig]:
    """按日常记忆整理的配置组装模型；返回服务、请求参数与提取器配置。"""
    config = load_app_config()
    if config.patchouli.llm is None:
        raise SystemExit("主配置没有 patchouli.llm，回放需要一个记忆整理模型")
    providers = ProviderRegistry(env_providers=provider_credentials_from_env())
    params = ModelRegistry(provider_registry=providers).resolve(config.patchouli.llm)
    return LiteLLMService(), params, config.patchouli.generation.extractor


async def complete_json(
    llm: BaseLLMService, params: LLMRequestParams, system: str, user: str
) -> Any:
    completion = await llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], params
    )
    return parse_llm_json(completion.text)


async def gather_limited(jobs: list[Callable[[], Awaitable[Any]]], concurrency: int) -> list[Any]:
    """按给定并发运行一组协程工厂，结果与输入同序。"""
    semaphore = asyncio.Semaphore(concurrency)

    async def run(job: Callable[[], Awaitable[Any]]) -> Any:
        async with semaphore:
            return await job()

    return await asyncio.gather(*(run(job) for job in jobs))


# ---------------------------------------------------------------- extract


async def extract(args: argparse.Namespace) -> None:
    """每份生成材料运行 ``repeats`` 次，一次一行；失败的一次 ``output`` 为 null 并带错误类名。"""
    rows = read_jsonl(args.settlements)[: args.limit or None]
    llm, params, extractor_config = librarian_model()
    builder = GenerationTranscriptBuilder()
    produce: Callable[[str], Awaitable[Any]]
    if args.prompt is None:
        extractor = create_extractor(extractor_config, llm, params)

        async def produce(transcript: str) -> Any:
            drafts = await extractor.extract(transcript)
            return {"memories": [d.model_dump(mode="json") for d in drafts]}

    else:
        system = args.prompt.read_text(encoding="utf-8")

        async def produce(transcript: str) -> Any:
            return await complete_json(llm, params, system, transcript)

    def job(row: dict[str, Any], run: int) -> Callable[[], Awaitable[dict[str, Any]]]:
        async def go() -> dict[str, Any]:
            # 早先导出的数据集没有轮次标识与段，读入时同样补齐。
            view = row["generation_view"]
            view = {**view, "turns": [upgrade_turn(t) for t in view.get("turns", [])]}
            transcript = builder.build_transcript(GenerationContext.model_validate(view))
            result: dict[str, Any] = {"id": row["id"], "run": run, "output": None}
            try:
                result["output"] = await produce(transcript)
            except (ModelProviderError, JSONParseError) as error:
                result["error"] = type(error).__name__
            return result

        return go

    jobs = [job(row, run) for row in rows for run in range(args.repeats)]
    results = await gather_limited(jobs, args.concurrency)
    write_jsonl(args.out, results)
    failed = sum(r["output"] is None for r in results)
    print(f"materials={len(rows)} runs={len(results)} failed={failed}")


# ---------------------------------------------------------------- classify


def render_batch(batch: list[dict[str, Any]]) -> str:
    parts = []
    for memory in batch:
        summary = f"摘要：{memory['summary']}\n" if memory.get("summary") else ""
        content = memory["content"][:CLASSIFY_CONTENT_CHARS]
        parts.append(
            f"### id: {memory['memory_id']}\n标题：{memory['title']}\n{summary}正文：\n{content}"
        )
    return "\n\n".join(parts)


async def classify(args: argparse.Namespace) -> None:
    """输出 ``{memory_id: [每次运行的结果或 null]}``；一批失败时这一批每条都记 null。"""
    memories = read_jsonl(args.memories)
    if args.source_intent:
        memories = [m for m in memories if m["source_intent"] == args.source_intent]
    memories = memories[: args.limit or None]
    system = args.prompt.read_text(encoding="utf-8")
    llm, params, _ = librarian_model()
    batches = [memories[k : k + args.batch] for k in range(0, len(memories), args.batch)]

    def job(batch: list[dict[str, Any]]) -> Callable[[], Awaitable[dict[str, Any]]]:
        async def go() -> dict[str, Any]:
            try:
                output = await complete_json(llm, params, system, render_batch(batch))
            except (ModelProviderError, JSONParseError):
                return {}
            results = output.get("results", []) if isinstance(output, dict) else []
            return {str(r.get("id")): r for r in results if isinstance(r, dict)}

        return go

    jobs = [job(batch) for _ in range(args.repeats) for batch in batches]
    outputs = await gather_limited(jobs, args.concurrency)
    answers: dict[str, list[Any]] = {m["memory_id"]: [] for m in memories}
    for k, output in enumerate(outputs):
        for memory in batches[k % len(batches)]:
            answers[memory["memory_id"]].append(output.get(memory["memory_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = sum(a is None for runs in answers.values() for a in runs)
    print(f"memories={len(memories)} runs={args.repeats} missing_answers={missing}")


# ---------------------------------------------------------------- agreement


def field_values(path: Path, field: str) -> dict[str, list[Any]]:
    """只保留每次运行都有结果的记忆，取出该字段在各次运行中的值。"""
    answers = json.loads(path.read_text(encoding="utf-8"))
    return {
        memory_id: [answer.get(field) for answer in runs]
        for memory_id, runs in answers.items()
        if runs and all(isinstance(answer, dict) for answer in runs)
    }


def majority(values: list[Any]) -> Any:
    """过半的取值；没有过半时为 None，因此与字段缺失（过半的值本身是 None）记法相同。"""
    value, count = collections.Counter(map(json.dumps, values)).most_common(1)[0]
    return json.loads(value) if count * 2 > len(values) else None


def agreement(args: argparse.Namespace) -> None:
    values = field_values(args.result, args.field)
    total = len(values)
    if total == 0:
        raise SystemExit("没有每次运行都有结果的记忆")
    stable = sum(len({json.dumps(v) for v in runs}) == 1 for runs in values.values())
    print(f"{args.field}: 每次运行都一致 {stable}/{total}（{100 * stable / total:.0f}%）")
    distribution = collections.Counter(
        json.dumps(majority(v), ensure_ascii=False) for v in values.values()
    )
    print("多数判定：" + "，".join(f"{k} {c}" for k, c in distribution.most_common()))
    swings = collections.Counter(
        " / ".join(sorted({json.dumps(v, ensure_ascii=False) for v in runs}))
        for runs in values.values()
        if len({json.dumps(v) for v in runs}) > 1
    )
    for swing, count in swings.most_common(args.top):
        print(f"  摇摆 {swing} ×{count}")
    if args.against is None:
        return
    other = field_values(args.against, args.against_field or args.field)
    pairs = [
        (majority(values[m]), majority(other[m]))
        for m in values.keys() & other.keys()
        if majority(values[m]) is not None and majority(other[m]) is not None
    ]
    same = sum(a == b for a, b in pairs)
    print(f"与 {args.against.name} 的多数判定相同：{same}/{len(pairs)}")
    differ = collections.Counter(
        f"{json.dumps(a, ensure_ascii=False)} → {json.dumps(b, ensure_ascii=False)}"
        for a, b in pairs
        if a != b
    )
    for pair, count in differ.most_common(args.top):
        print(f"  分歧 {pair} ×{count}")


# ---------------------------------------------------------------- turns


def turn_row(
    conversation_id: str, turn_id: str, outcome: str, events: list[TurnEvent]
) -> dict[str, Any]:
    return {
        "id": f"{conversation_id}/{turn_id}",
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "outcome": outcome,
        "events": [event.model_dump(mode="json") for event in events],
    }


def turns_claude_code(projects: Path, out: Path) -> None:
    """从头重读 ``projects`` 下的 Claude Code 会话记录，导出读取器切出的、有输出的轮次。

    只读会话记录。子 Agent 自己的会话记录（``subagents`` 目录）与连接器一样不读取；轮次经
    Import Bus 的输入模型转换，事件与服务端收到的相同，不合格的轮次计数后跳过。
    """
    sys.path.insert(0, str(CONNECTOR_ROOT))
    from connector.claude_code.reader import ClaudeCodeReader

    reader = ClaudeCodeReader()
    rows: list[dict[str, Any]] = []
    rejected = 0
    paths = [path for path in sorted(projects.rglob("*.jsonl")) if "subagents" not in path.parts]
    for path in paths:
        for turn in reader.read(str(path), 0):
            if not turn.events:
                continue
            try:
                imported = ImportedTurn(
                    source="claude-code",
                    external_conversation_id=turn.conversation_id,
                    external_turn_id=turn.turn_id,
                    user_message=turn.user_message,
                    outcome="cancelled" if turn.interrupted else "completed",
                    events=turn.events,
                )
            except ValidationError:
                rejected += 1
                continue
            rows.append(
                turn_row(
                    turn.conversation_id, turn.turn_id, imported.outcome, imported.to_turn_events()
                )
            )
    write_jsonl(out, rows)
    print(f"files={len(paths)} turns={len(rows)} rejected={rejected}")


def turns_snapshots(artifacts: Path, out: Path) -> None:
    """导出交互快照中的轮次，同一轮出现在多份快照中时只导出一次；解码不了的轮次计数后跳过。"""
    rows: dict[str, dict[str, Any]] = {}
    skipped = 0
    for path in artifacts.rglob("*.json"):
        if path.name.startswith("."):
            continue  # 存储自己的索引文件
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("artifact_type") != "interaction":
            continue
        for turn in record.get("turns", []):
            if turn["turn_id"] in rows:
                continue
            try:
                events = [TurnEvent.model_validate(event) for event in turn["turn_events"]]
                validate_turn_events(events)
            except (ValidationError, ValueError):
                skipped += 1
                continue
            rows[turn["turn_id"]] = turn_row(
                record["topic_id"], turn["turn_id"], turn["outcome"], events
            )
    write_jsonl(out, rows.values())
    print(f"turns={len(rows)} skipped={skipped}")


def read_turn_events(row: dict[str, Any]) -> list[TurnEvent]:
    return [TurnEvent.model_validate(event) for event in row["events"]]


def describe(values: list[int]) -> str:
    """中位数、P90 与最大值。"""
    if not values:
        return "-"
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return f"median={statistics.median(ordered):g} p90={p90} max={ordered[-1]}"


def segments(args: argparse.Namespace) -> None:
    """按每个上限切分全部轮次，打印每轮段数与每段响应数的分布。"""
    turns = [read_turn_events(row) for row in read_jsonl(args.turns)]
    indexed = [events for events in turns if any(e.response_index for e in events)]
    responses = [len({e.response_index for e in events if e.response_index}) for events in indexed]
    with_text = [
        sum(
            1
            for index in {e.response_index for e in events if e.response_index}
            if any(
                e.response_index == index
                and e.kind in ("assistant_message", "thought")
                and e.content.strip()
                for e in events
            )
        )
        for events in indexed
    ]
    print(f"turns={len(turns)} with_response_index={len(indexed)}")
    print(f"responses per turn: {describe(responses)}")
    print(
        "responses with text or thought: "
        f"{sum(with_text)}/{sum(responses)} ({100 * sum(with_text) / max(1, sum(responses)):.0f}%)"
    )
    for cap in args.max_responses:
        per_turn: list[int] = []
        per_segment: list[int] = []
        capped = 0
        for events in indexed:
            cut = segment_turn(events, max_responses=cap)
            per_turn.append(len(cut))
            for segment in cut:
                count = len(
                    {
                        e.response_index
                        for e in events
                        if e.response_index
                        and segment.first_sequence <= e.sequence <= segment.last_sequence
                    }
                )
                per_segment.append(count)
                capped += count == cap
        print(
            f"max_responses={cap}: segments per turn {describe(per_turn)}; "
            f"responses per segment {describe(per_segment)}; segments at cap {capped}"
        )


def turn_materials(
    rows: list[dict[str, Any]], max_chars: int
) -> list[tuple[str, GenerationContext]]:
    """把同一会话中连续的轮次拼成生成材料，渲染后不超过 ``max_chars``。

    单独一轮就超过上限时自成一份材料；超过上限三倍的轮次不进入材料，模拟不了一次结算。
    """
    builder = GenerationTranscriptBuilder()
    materials: list[tuple[str, GenerationContext]] = []
    for conversation_id, group in itertools.groupby(rows, key=lambda row: row["conversation_id"]):
        blocks: list[LogicalBlock] = []
        size = 0
        for row in group:
            block = LogicalBlock(
                turn=TurnRecord(
                    turn_id=row["turn_id"],
                    outcome=row["outcome"],
                    identity=REPLAY_IDENTITY,
                    turn_events=tuple(read_turn_events(row)),
                )
            )
            turn_size = len(builder.build_transcript(builder.build_context([block])))
            if turn_size > 3 * max_chars:
                continue
            if blocks and size + turn_size > max_chars:
                materials.append(
                    (f"{conversation_id}#{len(materials)}", builder.build_context(blocks))
                )
                blocks, size = [], 0
            blocks.append(block)
            size += turn_size
        if blocks:
            materials.append((f"{conversation_id}#{len(materials)}", builder.build_context(blocks)))
    return materials


def segment_labels(context: GenerationContext) -> list[str]:
    return [
        segment_label(position, number)
        for position, turn in enumerate(context.turns, 1)
        for number in range(1, len(turn.segments) + 1)
    ]


async def extract_turns(args: argparse.Namespace) -> None:
    """每份材料以现在的被动提取提示词运行 ``repeats`` 次，一次一行，记下模型原样给出的引用。

    材料写在输出旁的 ``materials.jsonl``，供 ``citations`` 核对引用。模型输出按生产的解析规则
    得到草稿（格式不对的引用被丢弃），再换算为事件区间；原样给出的引用另存一份，用来统计格式。
    """
    materials = turn_materials(read_jsonl(args.turns), args.max_chars)
    step = max(1, len(materials) // args.limit) if args.limit else 1
    chosen = materials[::step][: args.limit or None]
    llm, params, extractor_config = librarian_model()
    extractor = create_extractor(extractor_config, llm, params)
    if not isinstance(extractor, LLMMemoryExtractor):
        raise SystemExit("主配置的被动提取器没有启用")
    system = extractor.normal_system_prompt.format(
        format_instructions=extractor.passive_format_instructions
    )
    builder = GenerationTranscriptBuilder()
    transcripts = {material_id: builder.build_transcript(c) for material_id, c in chosen}
    write_jsonl(
        args.out.parent / "materials.jsonl",
        (
            {"id": material_id, "labels": segment_labels(c), "transcript": transcripts[material_id]}
            for material_id, c in chosen
        ),
    )

    def job(material_id: str, context: GenerationContext, run: int):
        async def go() -> dict[str, Any]:
            user = extractor.normal_user_prompt.format(transcript=transcripts[material_id])
            result: dict[str, Any] = {"id": material_id, "run": run, "memories": None}
            try:
                completion = await llm.complete(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    params,
                )
                raw = parse_llm_json(completion.text)
                drafts = ExtractionResult.model_validate(raw).memories
            except (ModelProviderError, JSONParseError, ValidationError) as error:
                result["error"] = type(error).__name__
                return result
            result["memories"] = [
                {
                    "title": draft.title,
                    "has_value": draft.has_value,
                    "raw_source_segments": raw_memory.get("source_segments"),
                    "source_segments": draft.source_segments,
                    "cited": [
                        c.model_dump() for c in context.cited_segments(draft.source_segments)
                    ],
                }
                for draft, raw_memory in zip(drafts, raw["memories"], strict=True)
            ]
            return result

        return go

    jobs = [job(i, c, run) for i, c in chosen for run in range(args.repeats)]
    results = await gather_limited(jobs, args.concurrency)
    write_jsonl(args.out, results)
    failed = sum(r["memories"] is None for r in results)
    print(f"materials={len(materials)} chosen={len(chosen)} runs={len(results)} failed={failed}")


_SEGMENT_LINE = re.compile(r"^\[Segment (\d+\.\d+)\]$", re.MULTILINE)


def segment_texts(transcript: str) -> dict[str, str]:
    """材料中每个段标签之后、下一个标签之前的文本。"""
    parts = _SEGMENT_LINE.split(transcript)
    return {label: text.strip() for label, text in zip(parts[1::2], parts[2::2], strict=True)}


def jaccard(a: set[str], b: set[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def citations(args: argparse.Namespace) -> None:
    """统计引用的格式、存在与多次运行的一致性，抽出若干条引用写到 ``--samples`` 供人工核对。"""
    materials = {row["id"]: row for row in read_jsonl(args.materials)}
    results = [row for row in read_jsonl(args.result) if row["memories"] is not None]
    memories = [m for row in results for m in row["memories"] if m["has_value"]]
    raw = [m["raw_source_segments"] for m in memories]
    raw_labels = [label for labels in raw if isinstance(labels, list) for label in labels]
    well_formed = [label for m in memories for label in m["source_segments"]]
    existing = [
        label
        for row in results
        for m in row["memories"]
        if m["has_value"]
        for label in m["source_segments"]
        if label in materials[row["id"]]["labels"]
    ]
    print(f"runs={len(results)} memories={len(memories)}")
    print(
        f"memories citing something: {sum(bool(m['source_segments']) for m in memories)}; "
        f"not a list: {sum(not isinstance(labels, list) for labels in raw)}"
    )
    print(
        f"labels given={len(raw_labels)} well_formed={len(well_formed)} existing={len(existing)} "
        f"(qualified {100 * len(existing) / max(1, len(raw_labels)):.1f}%)"
    )
    print(
        f"labels per citing memory: {describe([len(m['source_segments']) for m in memories if m['source_segments']])}"
    )
    by_material: dict[str, list[set[str]]] = collections.defaultdict(list)
    for row in results:
        by_material[row["id"]].append(
            {label for m in row["memories"] if m["has_value"] for label in m["source_segments"]}
        )
    pairs = [
        jaccard(a, b) for runs in by_material.values() for a, b in itertools.combinations(runs, 2)
    ]
    union = [set().union(*runs) for runs in by_material.values()]
    stable = [set.intersection(*runs) for runs in by_material.values() if runs]
    print(
        f"materials={len(by_material)} mean pairwise jaccard of cited labels="
        f"{statistics.fmean(pairs) if pairs else 0:.2f}; "
        f"labels cited in every run {sum(map(len, stable))}/{sum(map(len, union))}"
    )
    if args.samples is None:
        return
    lines = []
    for row in results[: args.top]:
        texts = segment_texts(materials[row["id"]]["transcript"])
        for m in row["memories"]:
            for label in m["source_segments"][:2]:
                excerpt = texts.get(label, "(不存在)")[:400].replace("\n", " ⏎ ")
                lines.append(f"- [{row['id']} run {row['run']}] {m['title']} → {label}: {excerpt}")
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.samples.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"samples written: {len(lines)}")


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在已有数据上离线回放记忆的生成与分类")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("export", help="从 Artifact 目录导出生成材料与建成的记忆")
    p.add_argument(
        "artifacts",
        type=Path,
        help="一个 Workspace 的 Artifact 目录（数据根下 .hivememory/artifacts/<workspace>）",
    )
    p.add_argument("out", type=Path, help="数据集目录")

    p = commands.add_parser("extract", help="把生成材料交给被动提取器或自定义提示词")
    p.add_argument("settlements", type=Path, help="每行一份生成材料，带 id 与 generation_view")
    p.add_argument("out", type=Path, help="输出的 jsonl")
    p.add_argument("--prompt", type=Path, help="自定义系统提示词；不给时用现在的被动提取器")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="只取前几份，0 为全部")
    p.add_argument("--concurrency", type=int, default=8)

    p = commands.add_parser("classify", help="把记忆按批交给分类提示词")
    p.add_argument("memories", type=Path, help="export 导出的 memories.jsonl")
    p.add_argument("prompt", type=Path, help="分类的系统提示词")
    p.add_argument("out", type=Path, help="输出的 json")
    p.add_argument("--source-intent", help="只取某种来源，例如被动结算的 SYSTEM")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="只取前几条，0 为全部")
    p.add_argument("--concurrency", type=int, default=8)

    p = commands.add_parser("agreement", help="统计 classify 结果中一个字段的一致性")
    p.add_argument("result", type=Path, help="classify 输出的 json")
    p.add_argument("field", help="要统计的字段，例如 type")
    p.add_argument("--against", type=Path, help="另一份 classify 输出，比较多数判定")
    p.add_argument("--against-field", help="另一份结果中对应的字段，默认同名")
    p.add_argument("--top", type=int, default=8, help="列出几组摇摆与分歧")

    p = commands.add_parser("turns-claude-code", help="从本机 Claude Code 会话记录导出轮次")
    p.add_argument("projects", type=Path, help="Claude Code 的 projects 目录，只读")
    p.add_argument("out", type=Path, help="输出的轮次 jsonl")

    p = commands.add_parser("turns-snapshots", help="从交互快照导出轮次")
    p.add_argument("artifacts", type=Path, help="一个 Workspace 的 Artifact 目录，只读")
    p.add_argument("out", type=Path, help="输出的轮次 jsonl")

    p = commands.add_parser("segments", help="按不同的每段响应上限统计切段")
    p.add_argument("turns", type=Path, help="导出的轮次 jsonl")
    p.add_argument("--max-responses", type=int, nargs="+", default=[4, 8, 12, 16])

    p = commands.add_parser("extract-turns", help="把轮次拼成材料交给被动提取，记下段引用")
    p.add_argument("turns", type=Path, help="导出的轮次 jsonl")
    p.add_argument("out", type=Path, help="输出的 jsonl；材料写在同目录的 materials.jsonl")
    p.add_argument("--max-chars", type=int, default=60000, help="一份材料渲染后的字符上限")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="均匀抽取几份材料，0 为全部")
    p.add_argument("--concurrency", type=int, default=8)

    p = commands.add_parser("citations", help="统计 extract-turns 结果中的引用")
    p.add_argument("materials", type=Path, help="extract-turns 写出的 materials.jsonl")
    p.add_argument("result", type=Path, help="extract-turns 输出的 jsonl")
    p.add_argument("--samples", type=Path, help="抽出的引用写到这个文件，供人工核对")
    p.add_argument("--top", type=int, default=10, help="从前几次运行中抽取引用")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        export(args.artifacts, args.out)
    elif args.command == "extract":
        asyncio.run(extract(args))
    elif args.command == "classify":
        asyncio.run(classify(args))
    elif args.command == "agreement":
        agreement(args)
    elif args.command == "turns-claude-code":
        turns_claude_code(args.projects, args.out)
    elif args.command == "turns-snapshots":
        turns_snapshots(args.artifacts, args.out)
    elif args.command == "segments":
        segments(args)
    elif args.command == "extract-turns":
        asyncio.run(extract_turns(args))
    else:
        citations(args)


if __name__ == "__main__":
    main()
