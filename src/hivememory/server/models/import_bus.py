"""Import Bus HTTP API 的 Request/Response 模型。"""

from pydantic import BaseModel, Field

from hivememory.system.services.import_bus import ImportedTurn, ImportTurnStatus


class ImportTurnRequest(ImportedTurn):
    """导入一轮已结束交互的完整记录；请求模型拒绝未知字段。

    ``agent_id`` 是连接器所代表的具体 Agent，只作为认证声明使用，不进入
    导入记录；缺失或为保留的 ``system`` 时由声明解析返回 400，因此这里
    不设为必填。
    """

    agent_id: str | None = Field(
        default=None,
        description="连接器所代表的具体 Agent ID；缺失或为 system 时返回 400",
    )

    def to_imported_turn(self) -> ImportedTurn:
        """去掉认证声明，交出只含导入内容的记录。"""
        return ImportedTurn.model_validate(self.model_dump(exclude={"agent_id"}))


class ImportTurnResponse(BaseModel):
    """导入结果：接纳、重复或因没有输出而跳过，并带推导出的交互标识。

    不暴露 Gateway 决定、话题或记忆上下文；可重试与不可重试的失败经错误
    信封的状态码区分。
    """

    status: ImportTurnStatus = Field(description="accepted、duplicate 或 skipped")
    interaction_id: str = Field(description="由认证结果与外部标识推导的稳定交互标识")


__all__ = ["ImportTurnRequest", "ImportTurnResponse"]
