"""Alice 子系统域事件常量 — 可通过桥接器上抛到 GlobalSystemBus 的领域事件。"""


class AliceEvents:
    RUN_STARTED = "alice.run.started"
    RUN_COMPLETED = "alice.run.completed"
