from hivememory.patchouli.tools.syscalls.file_io import sys_read_file, sys_write_file


class TestSysReadFile:
    """sys_read_file 函数直接测试"""

    def test_normal_read(self, tmp_path):
        (tmp_path / "hello.txt").write_text("Hello World", encoding="utf-8")
        result = sys_read_file({"path": "hello.txt"}, workspace=str(tmp_path))
        assert "Hello World" in result
        assert "<content>" in result

    def test_file_not_found(self, tmp_path):
        result = sys_read_file({"path": "missing.txt"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "not found" in result.lower()

    def test_path_traversal_blocked(self, tmp_path):
        result = sys_read_file({"path": "../../etc/passwd"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "denied" in result.lower() or "escape" in result.lower()

    def test_large_file_truncated(self, tmp_path):
        (tmp_path / "large.txt").write_text("A" * 200, encoding="utf-8")
        result = sys_read_file({"path": "large.txt"}, workspace=str(tmp_path), max_bytes=50)
        assert "Truncated" in result

    def test_missing_path_arg(self):
        result = sys_read_file({})
        assert "Error" in result
        assert "path" in result.lower()

    def test_binary_file_rejected(self, tmp_path):
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\x03" * 200)
        result = sys_read_file({"path": "binary.bin"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "binary" in result.lower()

    def test_subdirectory_read(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("print('hello')", encoding="utf-8")
        result = sys_read_file({"path": "src/main.py"}, workspace=str(tmp_path))
        assert "print('hello')" in result

# ========== Test 5: sys_write_file 直接调用 ==========

class TestSysWriteFile:
    """sys_write_file 函数直接测试"""

    def test_normal_write(self, tmp_path):
        result = sys_write_file(
            {"path": "output.txt", "content": "Hello"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "Hello"

    def test_path_traversal_blocked(self, tmp_path):
        result = sys_write_file(
            {"path": "../../evil.txt", "content": "bad"},
            workspace=str(tmp_path),
        )
        assert "Error" in result
        assert "denied" in result.lower() or "escape" in result.lower()

    def test_content_too_large(self, tmp_path):
        result = sys_write_file(
            {"path": "big.txt", "content": "A" * 200},
            workspace=str(tmp_path),
            max_bytes=50,
        )
        assert "Error" in result
        assert "too large" in result.lower()

    def test_missing_content(self, tmp_path):
        result = sys_write_file({"path": "empty.txt"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "content" in result.lower()

    def test_missing_path(self, tmp_path):
        result = sys_write_file({"content": "hello"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "path" in result.lower()

    def test_append_mode(self, tmp_path):
        (tmp_path / "append.txt").write_text("line1\n", encoding="utf-8")
        result = sys_write_file(
            {"path": "append.txt", "content": "line2\n", "mode": "append"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "line1\nline2\n"

    def test_invalid_mode(self, tmp_path):
        result = sys_write_file(
            {"path": "test.txt", "content": "data", "mode": "delete"},
            workspace=str(tmp_path),
        )
        assert "Error" in result
        assert "Invalid mode" in result

    def test_auto_create_parent_dirs(self, tmp_path):
        result = sys_write_file(
            {"path": "sub/dir/file.txt", "content": "nested"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    def test_overwrite_existing(self, tmp_path):
        (tmp_path / "exist.txt").write_text("old content", encoding="utf-8")
        result = sys_write_file(
            {"path": "exist.txt", "content": "new content"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "exist.txt").read_text(encoding="utf-8") == "new content"

# ========== Test 6: build_kernel_registry ==========
