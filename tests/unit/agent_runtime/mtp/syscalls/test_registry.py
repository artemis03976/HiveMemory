from hivememory.agent_runtime.mtp.syscalls.registry import build_kernel_registry
from hivememory.agent_runtime.mtp.syscalls.types import KernelSyscall


class TestKernelRegistry:
    """注册表构建测试"""

    def test_contains_all_syscalls(self):
        registry = build_kernel_registry()
        assert "sys_clock" in registry
        assert "sys_python_repl" in registry
        assert "sys_web_search" in registry
        assert "sys_read_file" in registry
        assert "sys_write_file" in registry

    def test_registry_types(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert isinstance(syscall, KernelSyscall), f"{name} is not KernelSyscall"

    def test_registry_descriptions(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert syscall.description, f"{name} has empty description"

    def test_registry_handlers_callable(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert callable(syscall.handler), f"{name} handler not callable"

    def test_custom_repl_timeout(self):
        registry = build_kernel_registry(python_repl_timeout=5)
        assert "sys_python_repl" in registry
