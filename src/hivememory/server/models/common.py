"""通用 Response 模型"""

from typing import Optional

from pydantic import BaseModel

from hivememory._version import __version__


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__


class ReadinessResponse(BaseModel):
    status: str = "ready"
    models_ready: bool = False
