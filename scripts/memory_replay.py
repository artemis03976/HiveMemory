"""在已有数据上离线回放记忆的生成与分类，比较同一份材料多次运行的结果。

回放只读已有数据，结果写到调用方给出的位置；不启动后端，不写任何记忆。模型按主配置的
``patchouli.llm`` 解析，被动提取器按 ``patchouli.generation.extractor`` 组装，与日常的
记忆整理相同；凭证与后端一样来自环境变量和 ``.env``。

子命令：

- ``export``：从一个 Workspace 的 Artifact 目录导出被动结算的生成材料，以及每条记忆建成时的
  类型与正文；Artifact 根目录下还留有迁移前的旧副本，所以不从根目录导出；
- ``extract``：把生成材料交给现在的被动提取器，或交给一份自定义的系统提示词，每份重复几次；
- ``classify``：把记忆按批交给一份分类提示词，每条重复几次；
- ``agreement``：统计 ``classify`` 结果中某个字段多次运行是否一致，以及两份结果的多数判定是否相同。

自定义提示词是一个文本文件，原样作为系统消息：``extract`` 的用户消息是渲染好的生成材料；
``classify`` 的用户消息是一批记忆，提示词要让模型输出
``{"results": [{"id": "<原样照抄的 id>", ...}]}``，``agreement`` 按其中的字段统计。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hivememory.config.app import load_app_config, provider_credentials_from_env
from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors.model import ModelProviderError
from hivememory.engines.generation.extractor import create_extractor
from hivememory.engines.generation.models import GenerationContext
from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams
from hivememory.infrastructure.llm.litellm_service import LiteLLMService
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

# 一批记忆中每条正文最多给模型看的字符数；分类只需要判断它是什么，不需要全文。
CLASSIFY_CONTENT_CHARS = 1500


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
    追加在末尾。
    """
    if "user_message" in turn:
        return turn
    entries = [e if e.startswith("- ") else f"- {e}" for e in turn.get("action_entries") or []]
    if turn.get("assistant_final_text"):
        entries.append(f"[Assistant]: {turn['assistant_final_text']}")
    return {
        "outcome": turn.get("outcome", "completed"),
        "user_message": turn.get("user_query", ""),
        "material_entries": entries,
        "identity": turn["identity"],
    }


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
            transcript = builder.build_transcript(
                GenerationContext.model_validate(row["generation_view"])
            )
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        export(args.artifacts, args.out)
    elif args.command == "extract":
        asyncio.run(extract(args))
    elif args.command == "classify":
        asyncio.run(classify(args))
    else:
        agreement(args)


if __name__ == "__main__":
    main()
