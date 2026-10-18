"""中立存储失败经 MTP 执行入口保留协议类别与安全呈现。"""

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.errors import StorageOfflineError, StorageReadError, StorageWriteError
from tests.helpers.workspace import make_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (StorageOfflineError, "mtp.system.storage_offline"),
        (StorageReadError, "mtp.system.storage_error"),
        (StorageWriteError, "mtp.system.storage_write_error"),
    ],
)
async def test_neutral_storage_failure_keeps_mtp_wire_category(error_type, code):
    """不同存储失败仍呈现既有系统错误，内部操作、地址和 cause 不交付给模型。"""

    async def fail(request):
        raise error_type(details={"operation": "internal-operation"}) from RuntimeError(
            "secret-host"
        )

    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs(["READ"]),
        submit_operation=fail,
        language="zh",
    )
    result = await KoakumaRuntime().execute_mtp("⟪ READ | known_alias ⟫", context)

    assert result.response_status == "error"
    assert f'code="{code}"' in result.formatted_response
    assert 'severity="system_fault"' in result.formatted_response
    assert "internal-operation" not in result.formatted_response
    assert "secret-host" not in result.formatted_response
