"""
HiveMemory - 帕秋莉感知引擎 (Perception Engine / MMU 算法层)

职责:
    提供无状态的短期记忆摄入纯算法（block 构造、token 估算、折叠阈值判断），
    一轮交互材料的解析与渲染（记忆生成视图、折叠摘要与 token 统计共用），
    折叠摘要与进展记录的写作者，以及沉淀材料与触发原因模型。

    话题路由、占用（lease）、沉淀、退役、删除与 LRU / 空闲维护由
    ``PerceptionFamiliar``（Patchouli 服务层）编排；本包不持有
    Store / Journal / Queue，也不导入 ``hivememory.patchouli.*``，保持
    patchouli -> engines 的单向依赖，可被其他 runtime 复用。

核心组件:
    - MemoryPerceptionEngine: 无状态摄入算法引擎（纯函数）
    - TriggerReason / TopicSettleMaterial: 沉淀的触发原因与交给生成的材料
    - FoldSummaryWriter: 折叠摘要的写作者
    - ProgressRecordWriter: 以旧进展记录与还没整理的交互写出进展记录的一条
"""

from hivememory.engines.perception.fold_summary import (
    FoldSummaryWriter,
    LLMFoldSummaryWriter,
    RuleFoldSummaryWriter,
    create_fold_summary_writer,
)
from hivememory.engines.perception.memory_perception_engine import (
    MemoryPerceptionEngine,
)
from hivememory.engines.perception.models import (
    LogicalBlock,
    TopicSettleMaterial,
    TriggerReason,
)
from hivememory.engines.perception.progress_record import ProgressRecordWriter

__all__ = [
    # 感知引擎（纯算法）
    "MemoryPerceptionEngine",
    # 数据模型
    "LogicalBlock",
    "TriggerReason",
    "TopicSettleMaterial",
    # 折叠摘要的写作者
    "FoldSummaryWriter",
    "RuleFoldSummaryWriter",
    "LLMFoldSummaryWriter",
    "create_fold_summary_writer",
    # 进展记录
    "ProgressRecordWriter",
]
