"""Chat 附件上传与解析的资源配置。

首轮限制取自计划 7.7 节的建议默认值；W1-A 的上传接收与 W1-B parser
共用同一个 RAW 字节数上限，避免"上传合规但解析必然超限"的配置漂移。
"""

from pydantic import BaseModel, ConfigDict, Field


class AttachmentsConfig(BaseModel):
    """附件上传硬限制。

    ``max_raw_bytes`` 是单个附件原始内容的硬上限，应用层按实际读取
    字节数判断，不信任 ``Content-Length``；``max_display_name_length``
    约束规范化后文件名的长度，文件名只作为展示用途。
    """

    max_raw_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_display_name_length: int = Field(default=200, ge=1)

    model_config = ConfigDict(extra="ignore")
