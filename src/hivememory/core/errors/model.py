"""模型解析与模型调用的共享错误。"""


class ModelNotFoundError(Exception):
    """注册表中找不到指定 ID 的模型。

    由模型解析端口的实现方（System 模型注册表）抛出，Agent 执行侧据此区分
    "模型不存在"与其他解析失败；定义在 core 以免执行侧依赖 System。
    """


class ModelProviderError(Exception):
    """模型提供商拒绝或未能完成请求。

    由 LLM 端口的适配器在 SDK 边界产生，调用方只按本类型识别提供商失败，
    不接触 SDK 异常。由 SDK 异常转换而来时原始异常保留在 cause 链中；提供商
    响应缺少所需内容时没有原始异常。程序错误与取消不会被转换为本类型。
    """


class ModelContextWindowExceededError(ModelProviderError):
    """提供商报告上下文超窗；异常正文仅保留在内部异常链。

    作为提供商失败的一种，按提供商失败处理的调用方无需单独捕获；需要区分
    超窗的调用方（如 CPU 失败原因投影）必须先于 ``ModelProviderError`` 判断。
    """


__all__ = ["ModelContextWindowExceededError", "ModelNotFoundError", "ModelProviderError"]
