"""R3-R4 迁移期兼容入口；生产代码应直接使用 RunScheduler。"""

from hivememory.alice.runtime.agent.run_scheduler import RunScheduler

RunDriver = RunScheduler

__all__ = ["RunDriver"]
