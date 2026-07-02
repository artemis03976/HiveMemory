"""
HiveMemory 提供商凭证注册表

ProviderRegistry 管理所有 LLM 提供商（provider）的凭证（api_key/api_base）。
持久化存储在 configs/providers.secrets.yaml（gitignored）。

凭证优先级（由高到低）：
  1. 环境变量 HIVEMEMORY__PROVIDERS__<NAME>__* — 适合 CI/CD 注入，不可被覆盖
  2. providers.secrets.yaml — 用户通过前端 UI 管理，运行时 CRUD 立即生效
  3. 无凭证（None）— 由 litellm 自行读取对应环境变量兜底

设计原则：
  - 写操作只写 providers.secrets.yaml（不修改环境变量）
  - 读操作先查 env（只读），再查 yaml；env 优先保证部署覆盖语义
  - 修改立即生效，无需重启服务
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from hivememory.system.config.shared import ProviderCredentials

logger = logging.getLogger(__name__)


class ProviderNotFoundError(Exception):
    """注册表中找不到指定名称的提供商"""


class ProviderRegistry:
    """
    提供商凭证注册表

    职责：
    - 从 configs/providers.secrets.yaml 加载用户管理的凭证
    - 合并环境变量注入的凭证（env 优先于 yaml）
    - 提供 CRUD 操作（list / get / upsert / delete）
    - 将 yaml 层变更原子性地持久化到文件

    不变量：
    - env 层只读：API 层写操作只改 yaml 层，不覆盖 env
    - get() 始终先查 env 层，再查 yaml 层
    """

    def __init__(
        self,
        secrets_path: Optional[Path] = None,
        env_providers: Optional[Dict[str, ProviderCredentials]] = None,
    ):
        """
        Args:
            secrets_path: providers.secrets.yaml 路径。None 时使用项目默认路径。
            env_providers: 环境变量注入的凭证（来自 SharedConfig.providers）。
                          这层优先级最高，API 写操作不会影响它。
        """
        self._path = secrets_path or self._default_path()
        # env 层：只读，优先级最高
        self._env: Dict[str, ProviderCredentials] = {
            name.lower(): cred for name, cred in (env_providers or {}).items()
        }
        # yaml 层：用户通过 UI 管理，可读写
        self._yaml: Dict[str, ProviderCredentials] = {}
        self._load()

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------

    @staticmethod
    def _default_path() -> Path:
        """解析默认凭证文件路径（configs/providers.secrets.yaml）"""
        # provider_registry.py 位于 src/hivememory/system/，向上三级到达项目根目录
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "configs" / "providers.secrets.yaml"

    # ------------------------------------------------------------------
    # 加载 & 持久化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """从 providers.secrets.yaml 加载 yaml 层凭证。文件不存在时以空表启动。"""
        if not self._path.exists():
            logger.info(f"提供商凭证文件未找到: {self._path}，使用空 yaml 层启动")
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_providers: Dict[str, dict] = data.get("providers") or {}
            for name, raw in raw_providers.items():
                if not isinstance(raw, dict):
                    continue
                try:
                    self._yaml[name.lower()] = ProviderCredentials(**raw)
                except Exception as e:
                    logger.error(f"跳过无效提供商凭证 '{name}': {e}")

            logger.info(
                f"已加载 {len(self._yaml)} 个提供商凭证（yaml 层，来自 {self._path}）"
            )
        except Exception as e:
            logger.error(f"加载提供商凭证文件失败: {e}")

    def _save(self) -> None:
        """
        将 yaml 层凭证原子性地写入文件。
        env 层不参与持久化（来自环境变量，由部署侧管理）。
        """
        payload = {
            "providers": {
                name: cred.model_dump(mode="json")
                for name, cred in self._yaml.items()
            }
        }

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent,
            suffix=".tmp",
            prefix="providers_secrets_",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                # 文件头注释，提醒用户勿手动提交
                f.write(
                    "# HiveMemory 提供商凭证（由系统自动维护，已加入 .gitignore）\n"
                    "# 参考 providers.secrets.example.yaml 了解字段说明\n\n"
                )
                yaml.safe_dump(
                    payload,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            os.replace(tmp_path, self._path)
            logger.debug(f"提供商凭证已持久化（{len(self._yaml)} 条 yaml 层记录）")
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ProviderCredentials]:
        """
        获取提供商凭证。优先返回 env 层（不可覆盖），其次 yaml 层。

        Args:
            name: 提供商名称（大小写不敏感）

        Returns:
            ProviderCredentials，或 None（两层均无此提供商）
        """
        lower = name.lower()
        return self._env.get(lower) or self._yaml.get(lower)

    def list_all(self) -> List[Tuple[str, ProviderCredentials, bool]]:
        """
        返回所有已知提供商的凭证列表。

        合并 env 层与 yaml 层，env 层优先。
        返回值：List of (name, credentials, is_from_env)
          - is_from_env=True 表示此提供商来自环境变量（只读）
        """
        merged: Dict[str, Tuple[ProviderCredentials, bool]] = {}
        # 先填 yaml 层（低优先）
        for name, cred in self._yaml.items():
            merged[name] = (cred, False)
        # 再填 env 层（高优先）
        for name, cred in self._env.items():
            merged[name] = (cred, True)
        return [(name, cred, from_env) for name, (cred, from_env) in merged.items()]

    def has_provider(self, name: str) -> bool:
        """判断指定提供商是否存在（env 层或 yaml 层均算）"""
        lower = name.lower()
        return lower in self._env or lower in self._yaml

    # ------------------------------------------------------------------
    # 写入接口（仅操作 yaml 层）
    # ------------------------------------------------------------------

    def upsert(self, name: str, credentials: ProviderCredentials) -> None:
        """
        创建或更新 yaml 层中的提供商凭证。

        注意：若同名 provider 已存在于 env 层，此操作仍会写入 yaml 层，
        但 get() 返回时 env 层优先，yaml 层的值在 env 存在时不会被使用。

        Args:
            name: 提供商名称（统一转小写存储）
            credentials: 新凭证
        """
        lower = name.lower()
        self._yaml[lower] = credentials
        self._save()
        logger.info(f"已写入提供商凭证: {lower} (yaml 层)")

    def delete(self, name: str) -> None:
        """
        从 yaml 层删除提供商凭证。

        若提供商只存在于 env 层，此操作无法删除（env 层只读）。

        Args:
            name: 提供商名称

        Raises:
            ProviderNotFoundError: yaml 层中不存在此提供商
        """
        lower = name.lower()
        if lower not in self._yaml:
            raise ProviderNotFoundError(
                f"提供商 '{lower}' 不在 yaml 层中（无法删除来自环境变量的凭证）"
            )
        del self._yaml[lower]
        self._save()
        logger.info(f"已删除提供商凭证: {lower} (yaml 层)")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """返回 yaml 层记录数（不含 env 层）"""
        return len(self._yaml)

    def __repr__(self) -> str:
        return (
            f"ProviderRegistry("
            f"yaml_providers={list(self._yaml.keys())}, "
            f"env_providers={list(self._env.keys())}, "
            f"path={self._path})"
        )