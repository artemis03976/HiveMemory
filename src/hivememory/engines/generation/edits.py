"""
HiveMemory - 编辑集的应用

馆长的修订、主动 UPDATE 与查重提名的合并都以编辑集改一条已有记忆：正文、标题与摘要的片段
替换，正文的追加与整篇改写，标签的逐个增删。整篇重写在多次修订中会让细节漂移；片段替换要求
原文逐字出现且只出现一次，模型因此有了“只改变化的那一处”的选项。

检索与查重的向量只由标题、类型、标签与摘要编译，不含正文。这几项只能片段替换、逐个增删，
类型不变，索引因此只会有限度地变化；不再另外比较修改前后的向量。

编辑集不能应用是模型输出被拒的一种；WRITE 的草稿解析不了是另一种。两者都以
``InvalidOutputError`` 表达，生成引擎据此带着上一次的输出与说明再请模型一次。
"""

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
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

_INDEX_LIMITS = {"title": MEMORY_TITLE_MAX_LENGTH, "summary": MEMORY_SUMMARY_MAX_LENGTH}


class InvalidOutputError(ValueError):
    """模型的输出不能用：解析不了、结构不对或不能落地。说明原样交回模型，供它重试时改正。"""


class EditSetError(InvalidOutputError):
    """编辑集不能以给定的记忆为底应用；说明指出第几处编辑与原因，原样交回模型。"""


# 上一次的模型输出与它被拒的说明；WRITE、修订与合并带着它再请模型输出一次。
type OutputRetry = tuple[str, str]


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


__all__ = [
    "EditSetError",
    "InvalidOutputError",
    "OutputRetry",
    "apply_edit_set",
    "describe_validation_error",
    "parse_edit_set",
    "rebase_edit_set",
]
