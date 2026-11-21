"""
HiveMemory - 编辑集的应用与漂移拦截

馆长的修订、主动 UPDATE 与查重提名的合并都以编辑集改一条已有记忆：正文、标题与摘要的片段
替换，正文的追加与整篇改写，标签的逐个增删。整篇重写在多次修订中会让细节漂移；片段替换要求
原文逐字出现且只出现一次，模型因此有了“只改变化的那一处”的选项。

检索与查重的向量只由标题、类型、标签与摘要编译，不含正文。修订让这几项一起变化时，落地之前
比较修改前后两版的稠密向量：余弦相似度低于阈值说明这条记忆被改成了另一件事，拒绝这次修订。
"""

import math

from pydantic import ValidationError

from hivememory.core.models import MemoryAtom
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH, MEMORY_TITLE_MAX_LENGTH
from hivememory.core.models.provenance import normalize_contributing_agent_ids
from hivememory.engines.generation.models import (
    AppendEdit,
    MemoryEditSet,
    ReplaceEdit,
    RewriteEdit,
)
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget
from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

_compiler = MemoryCompiler()

_INDEX_LIMITS = {"title": MEMORY_TITLE_MAX_LENGTH, "summary": MEMORY_SUMMARY_MAX_LENGTH}


class EditSetError(ValueError):
    """编辑集不能以给定的记忆为底应用；说明指出第几处编辑与原因，原样交回模型。"""


class IndexDriftError(Exception):
    """修改前后两版索引的稠密相似度低于阈值：这条记忆的标题、标签或摘要变化过大。"""

    def __init__(self, similarity: float, min_similarity: float) -> None:
        super().__init__(
            f"index similarity {similarity:.3f} is below {min_similarity:.3f}: the title, tags "
            "or summary changed too much"
        )
        self.similarity = similarity
        self.min_similarity = min_similarity


# 上一次的模型输出与它被拒的原因；修订与合并带着它再请模型写一次编辑集。
type EditSetRetry = tuple[str, EditSetError | IndexDriftError]


def parse_edit_set(text: str) -> MemoryEditSet:
    """把模型输出解析为编辑集。

    说明文字是英文的字段路径与原因，与参数校验错误的写法相同，不随提示词的语言变化。

    Raises:
        EditSetError: 输出不是一个 JSON 对象，或不符合编辑集的结构
    """
    try:
        data = parse_llm_json(text)
    except JSONParseError as error:
        raise EditSetError("the output is not a JSON object") from error
    if not isinstance(data, dict):
        raise EditSetError("the output is not a JSON object")
    try:
        return MemoryEditSet.model_validate(data)
    except ValidationError as error:
        raise EditSetError(describe_validation_error(error)) from error


def describe_validation_error(error: ValidationError) -> str:
    """参数校验错误的简短说明：每处错误的字段路径与原因。"""
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )


def apply_edit_set(memory: MemoryAtom, edit_set: MemoryEditSet) -> MemoryAtom:
    """以 ``memory`` 为底应用编辑集，返回修改后的副本，不修改输入。

    编辑按顺序应用，后一处在前一处的结果上查找原文。标题与摘要越界时拒绝而不是截断：
    截断会让修订静默地丢掉一部分。类型、访问策略、alias、版本与来源都沿用原记忆，
    版本与内容时间由提交边界推进。

    Raises:
        EditSetError: 原文没有逐字出现或出现不止一次，标题为空，标题或摘要越界，标签的
            增删与现有标签不符，或者应用之后什么也没有改
    """
    texts = {
        "content": memory.payload.content,
        "title": memory.index.title,
        "summary": memory.index.summary,
    }
    for number, edit in enumerate(edit_set.edits, 1):
        match edit:
            case ReplaceEdit(field=field, old=old, new=new):
                texts[field] = _replace_once(texts[field], old, new, number=number, field=field)
            case AppendEdit(text=text):
                body = texts["content"].rstrip()
                texts["content"] = f"{body}\n\n{text.strip()}" if body else text.strip()
            case RewriteEdit(text=text):
                texts["content"] = text

    for name, limit in _INDEX_LIMITS.items():
        texts[name] = texts[name].strip()
        if len(texts[name]) > limit:
            raise EditSetError(
                f"the {name} would be {len(texts[name])} characters long; the limit is {limit}"
            )
    if not texts["title"]:
        raise EditSetError("the title would be empty")

    revised = memory.model_copy(deep=True)
    revised.index.title = texts["title"]
    revised.index.summary = texts["summary"]
    revised.index.tags = _edited_tags(memory.index.tags, edit_set)
    revised.payload.content = texts["content"]
    if revised.index == memory.index and revised.payload.content == memory.payload.content:
        raise EditSetError("the edit set changes nothing")
    return revised


def rebase_edit_set(
    revised: MemoryAtom, edit_set: MemoryEditSet, current: MemoryAtom
) -> MemoryAtom:
    """把产生 ``revised`` 的编辑集重放到 ``current`` 上，用于提交前基础的版本变了的时候。

    期间别处的修改只要没有碰到编辑引用的原文，就在结果中保留；内容贡献者取两边的并集。
    空编辑集（合并时被吸收的内容已被覆盖）重放的结果就是当前版本。

    Raises:
        EditSetError: 编辑引用的原文在当前版本中已经对不上
    """
    rebased = (
        current.model_copy(deep=True) if edit_set.is_empty else apply_edit_set(current, edit_set)
    )
    rebased.meta.provenance.contributing_agent_ids = normalize_contributing_agent_ids(
        [
            *current.meta.provenance.contributing_agent_ids,
            *revised.meta.provenance.contributing_agent_ids,
        ]
    )
    return rebased


def _replace_once(text: str, old: str, new: str, *, number: int, field: str) -> str:
    """把 ``text`` 中唯一的一处 ``old`` 换成 ``new``；重叠的出现也算作不唯一。"""
    first = text.find(old)
    if first < 0:
        raise EditSetError(
            f"edit {number} (replace in {field}): the old text does not occur in the current "
            f"{field}; copy it verbatim from the memory"
        )
    if text.find(old, first + 1) >= 0:
        raise EditSetError(
            f"edit {number} (replace in {field}): the old text occurs more than once in the "
            f"current {field}; include more surrounding text so that it occurs exactly once"
        )
    return text[:first] + new + text[first + len(old) :]


def _edited_tags(current: list[str], edit_set: MemoryEditSet) -> list[str]:
    """逐个增删标签；比较按索引层的规范化（小写、去首尾空白）进行。"""
    removed = [_normalize_tag(tag) for tag in edit_set.remove_tags]
    added = [tag for tag in (_normalize_tag(tag) for tag in edit_set.add_tags) if tag]
    for tag in removed:
        if tag not in current:
            raise EditSetError(f"remove_tags: '{tag}' is not one of the current tags {current}")
    for tag in added:
        if tag in current:
            raise EditSetError(f"add_tags: '{tag}' is already one of the current tags")
    tags = [tag for tag in current if tag not in removed] + added
    if removed and not tags:
        raise EditSetError("remove_tags would leave the memory without tags")
    return tags


def _normalize_tag(tag: str) -> str:
    return tag.lower().strip()


class IndexDriftGuard:
    """比较一次修改前后两版索引的稠密向量，余弦相似度低于阈值时拒绝。

    向量输入与中期库编码记忆时相同（``DENSE_EMBEDDING`` 的编译文本），由同一个嵌入服务编码，
    因此相似度反映检索与查重实际看到的变化。比较的是这一次修改的前后两版，不与最初的版本比较。
    """

    def __init__(self, embedding_service: BaseEmbeddingService, *, min_similarity: float) -> None:
        self._embedding_service = embedding_service
        self.min_similarity = min_similarity

    async def check(self, before: MemoryAtom, after: MemoryAtom) -> float:
        """返回两版索引的稠密相似度；只改了正文时两段编译文本相同，相似度为 1，不编码。

        Raises:
            IndexDriftError: 相似度低于阈值
        """
        before_text = _compiler.compile(before, MemoryCompileTarget.DENSE_EMBEDDING).text
        after_text = _compiler.compile(after, MemoryCompileTarget.DENSE_EMBEDDING).text
        if before_text == after_text:
            return 1.0
        vectors = await self._embedding_service.encode(dense_texts=[before_text, after_text])
        similarity = _cosine(*_two_vectors(vectors))
        if similarity < self.min_similarity:
            raise IndexDriftError(similarity, self.min_similarity)
        return similarity


def _two_vectors(output: object) -> tuple[list[float], list[float]]:
    """批量编码两段文本的结果；嵌入服务推理失败时退化为空向量，这里显式失败而不是放行。"""
    if (
        not isinstance(output, list)
        or len(output) != 2
        or not all(isinstance(vector, list) and vector for vector in output)
    ):
        raise RuntimeError("嵌入服务没有为修改前后的索引返回两条稠密向量")
    return output[0], output[1]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


__all__ = [
    "EditSetError",
    "EditSetRetry",
    "IndexDriftError",
    "IndexDriftGuard",
    "apply_edit_set",
    "describe_validation_error",
    "parse_edit_set",
    "rebase_edit_set",
]
