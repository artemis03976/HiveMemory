"""Chat 附件的组件级配置（计划 7.7 / 10.4 节）。

按"一个组件一份 config"的既有惯例拆分：``AttachmentParserConfig`` 承载
W1-B 确定性解析的资源限制，``AttachmentCompilerConfig`` 承载 W1-E 附件
编译的预算。``max_raw_bytes`` 由上传接收（W1-A）与 parser（W1-B）共用
同一个值（计划 7.7 节），避免"上传合规但解析必然超限"的配置漂移。

首轮限制取自计划的建议默认值，实现时用边界样例校准。
"""

from pydantic import BaseModel, ConfigDict, Field

_MIB = 1024 * 1024


class AttachmentParserConfig(BaseModel):
    """确定性解析（W1-B）的资源限制。

    解析输出、解压和结构上限独立于 RAW 输入上限计算，因此上传合规的
    文件仍可能在解析阶段因输出超限进入 FAILED，这不属于上传状态回滚。
    """

    # 与上传接收（W1-A）共用的 RAW 字节数硬上限。
    max_raw_bytes: int = Field(default=10 * _MIB, ge=1)
    # 提取正文 UTF-8 大小上限，在解码/提取过程中累计。
    max_extracted_text_bytes: int = Field(default=8 * _MIB, ge=1)
    # 完整 canonical 内容大小上限（包含 locator/warning）。
    max_canonical_content_bytes: int = Field(default=16 * _MIB, ge=1)
    # locator 数量上限，超限整体失败，不静默截断。
    max_locator_count: int = Field(default=50_000, ge=1)

    # DOCX 包成员数上限（ZIP 目录预检）。
    max_docx_members: int = Field(default=1_024, ge=1)
    # DOCX 单成员声明解压大小上限。
    max_docx_member_uncompressed_bytes: int = Field(default=16 * _MIB, ge=1)
    # DOCX 全包声明解压大小上限。
    max_docx_package_uncompressed_bytes: int = Field(default=32 * _MIB, ge=1)
    # DOCX 声明压缩比上限，覆盖零压缩长度异常。
    max_docx_compression_ratio: int = Field(default=100, ge=1)
    # XML 深度上限（流式遍历累计）。
    max_xml_depth: int = Field(default=128, ge=1)
    # XML 节点数上限（流式遍历累计）。
    max_xml_nodes: int = Field(default=200_000, ge=1)

    # 单次解析预算秒数（可注入单调时钟检查）。
    parse_budget_seconds: float = Field(default=5.0, gt=0)

    model_config = ConfigDict(extra="ignore")


class AttachmentCompilerConfig(BaseModel):
    """附件编译（W1-E）的预算。

    多附件严格按用户选择顺序处理；单附件超出字符预算或 chunk 数上限时
    在 locator 边界保留前部完整内容并声明 truncated；合计超出总预算时
    跳过剩余附件。
    """

    # 单附件参与编译的最大正文字符数（Unicode 码点）。
    max_attachment_chars: int = Field(default=24_000, ge=1)
    # 内部 chunk 的最大字符数；chunk 只是编译内部处理步骤，不外露。
    max_chunk_chars: int = Field(default=4_000, ge=1)
    # 单附件保留的最大 chunk 数。
    max_chunks_per_attachment: int = Field(default=12, ge=1)
    # 全部附件 section 合计的最大字符数。
    max_total_context_chars: int = Field(default=48_000, ge=1)

    model_config = ConfigDict(extra="ignore")
