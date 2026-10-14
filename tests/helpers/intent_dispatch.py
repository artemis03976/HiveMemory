"""仅在未测试 Patchouli 生成链的场景中替代其公共接纳边界。"""

from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntentSubmissionResult
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmissionStatus


class IntentAdmissionStub:
    """记录真实 actor 提交的载荷，正式生成与结算由各测试显式驱动。"""

    def __init__(self) -> None:
        self.submissions = []

    async def submit(self, *, intent, identity_scope):
        """模拟队列已接纳；不冒充正式生成或结算事件。"""
        self.submissions.append((intent, identity_scope))
        return MemoryIntentSubmissionResult(
            status=MemoryGenerationAdmissionStatus.ACCEPTED,
            task_id=f"active:{intent.intent_id}",
            task_status="pending",
        )


def install_intent_admission(bus):
    """由测试装配声明 Patchouli 位于边界外，返回可检查的提交记录。"""
    stub = IntentAdmissionStub()
    bus.register(PatchouliRoutes.MEMORY_INTENT_SUBMIT, stub.submit)
    return stub
