"""模型解析端口的共享错误。"""


class ModelNotFoundError(Exception):
    """注册表中找不到指定 ID 的模型。

    由模型解析端口的实现方（System 模型注册表）抛出，Agent 执行侧据此区分
    "模型不存在"与其他解析失败；定义在 core 以免执行侧依赖 System。
    """


__all__ = ["ModelNotFoundError"]
