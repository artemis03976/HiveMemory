"""执行凭据的进程内契约：禁止序列化为可持久化载体。"""

import pickle

import pytest

from hivememory.workspace.contracts import ExecutionCredential


def test_execution_credential_cannot_be_serialized():
    """序列化必须明确失败，不能产生可在其他运行中重建的凭据。"""
    with pytest.raises(TypeError, match="Execution credentials cannot be serialized"):
        pickle.dumps(ExecutionCredential())
