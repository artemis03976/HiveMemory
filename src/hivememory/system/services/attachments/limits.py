"""解析资源限制与协作式预算（计划 7.7 节）。

这些是进程内输入和工作量限制，不是进程内存硬配额；协作式检查不能强制
中止卡住的库调用。默认值取自计划的建议值，实现时通过 System 配置
（``AttachmentsConfig``）覆盖并使用边界样例校准。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from hivememory.system.services.attachments.errors import RESOURCE_LIMIT, AttachmentParseError

_MIB = 1024 * 1024


@dataclass(frozen=True)
class ParseLimits:
    """一次解析固定的工作量限制；同一服务运行期内不热替换。"""

    #: RAW 字节输入上限（与 W1-A 上传限制共用同一个值，W1-B 再核对）。
    max_raw_bytes: int = 10 * _MIB
    #: 提取正文 UTF-8 大小上限，在解码/提取过程中累计。
    max_extracted_text_bytes: int = 8 * _MIB
    #: 完整 canonical 内容大小上限（包含 locator/warning）。
    max_canonical_content_bytes: int = 16 * _MIB
    #: locator 数量上限，超限整体失败，不静默截断。
    max_locator_count: int = 50_000
    #: DOCX 包成员数上限（ZIP 目录预检）。
    max_docx_members: int = 1_024
    #: DOCX 单成员声明解压大小上限。
    max_docx_member_uncompressed_bytes: int = 16 * _MIB
    #: DOCX 全包声明解压大小上限。
    max_docx_package_uncompressed_bytes: int = 32 * _MIB
    #: DOCX 声明压缩比上限，覆盖零压缩长度异常。
    max_docx_compression_ratio: int = 100
    #: XML 深度上限（流式遍历累计）。
    max_xml_depth: int = 128
    #: XML 节点数上限（流式遍历累计）。
    max_xml_nodes: int = 200_000
    #: 单次解析预算秒数（可注入单调时钟检查）。
    parse_budget_seconds: float = 5.0


class ParseBudget:
    """协作式解析预算。

    在读取和遍历检查点调用 :meth:`check`；超时属于运行结果，不承诺
    跨机器每次都发生相同失败。时钟可注入以便测试使用小额预算。
    """

    def __init__(
        self,
        limits: ParseLimits,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._seconds = limits.parse_budget_seconds
        self._clock = clock
        self._started_at = clock()

    def check(self) -> None:
        """在检查点核对剩余预算；超限抛出 ``resource_limit`` 受控失败。"""
        if self._clock() - self._started_at > self._seconds:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件解析耗时超过预算，请缩小文件后重新上传",
                params={"reason": "parse_budget_exceeded"},
            )


__all__ = ["ParseBudget", "ParseLimits"]
