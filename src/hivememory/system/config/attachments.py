"""Chat 附件上传与解析的资源配置。

限制默认值取自计划 7.7 节的建议值；W1-A 的上传接收与 W1-B parser
共用同一个 RAW 字节数上限，避免"上传合规但解析必然超限"的配置漂移。
"""

from pydantic import BaseModel, ConfigDict, Field

_MIB = 1024 * 1024


class AttachmentsConfig(BaseModel):
    """附件上传与解析的硬限制。

    ``max_raw_bytes`` 是单个附件原始内容的硬上限，应用层按实际读取
    字节数判断，不信任 ``Content-Length``；``max_display_name_length``
    约束规范化后文件名的长度，文件名只作为展示用途。解析阶段限制
    （输出/解压/结构/预算）独立于 RAW 输入上限计算，上传合规的文件
    仍可能在解析阶段因输出超限进入 FAILED。
    """

    max_raw_bytes: int = Field(default=10 * _MIB, ge=1)
    max_display_name_length: int = Field(default=200, ge=1)

    # ---- 解析阶段限制（计划 7.7 节，W1-B parser 使用） ----
    max_extracted_text_bytes: int = Field(default=8 * _MIB, ge=1)
    max_canonical_content_bytes: int = Field(default=16 * _MIB, ge=1)
    max_locator_count: int = Field(default=50_000, ge=1)
    max_docx_members: int = Field(default=1_024, ge=1)
    max_docx_member_uncompressed_bytes: int = Field(default=16 * _MIB, ge=1)
    max_docx_package_uncompressed_bytes: int = Field(default=32 * _MIB, ge=1)
    max_docx_compression_ratio: int = Field(default=100, ge=1)
    max_xml_depth: int = Field(default=128, ge=1)
    max_xml_nodes: int = Field(default=200_000, ge=1)
    parse_budget_seconds: float = Field(default=5.0, gt=0)

    model_config = ConfigDict(extra="ignore")
