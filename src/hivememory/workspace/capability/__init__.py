"""workspace 能力层共享 backing；root 与 actor API 从各自子包取得。"""

from hivememory.workspace.capability.backing import BusCanonicalReadBackend

__all__ = ["BusCanonicalReadBackend"]
