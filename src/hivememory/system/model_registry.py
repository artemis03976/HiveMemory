"""
HiveMemory 模型注册表服务

ModelRegistry 是管理所有可用 LLM 模型的单一数据源（SSOT）。
持久化存储在 configs/models.yaml，支持运行时 CRUD 操作。

使用方式：
    registry = ModelRegistry()
    params = registry.resolve(ModelSelection(model_id="deepseek-chat", temperature=0.2))
    # model_id='default' 选取默认模型；上下文窗口与展示名从模型定义读取
    model = registry.select_model("default")
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from hivememory.config.shared import ProviderCredentials
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models.model_definition import ModelDefinition, ModelSelection
from hivememory.infrastructure.llm.base import LLMRequestParams

if TYPE_CHECKING:
    from hivememory.system.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


class DuplicateModelIdError(Exception):
    """尝试添加已存在的模型 ID"""


class ModelRegistry:
    """
    模型注册表 — 管理系统中所有可用的 LLM 模型

    职责：
    - 从 configs/models.yaml 加载模型列表
    - 提供 CRUD 操作（list / get / add / update / delete）
    - 把模型 ID 解析为 LLM 调用的请求参数（Alice 与组件共用）
    - 将变更原子性地持久化到 YAML 文件

    不变量：
    - 注册表中有且仅有一条记录的 is_default=True
    - 若注册表为空或没有默认模型，resolve("default") 返回列表中第一条记录
    """

    def __init__(
        self,
        registry_path: Path | None = None,
        provider_registry: Optional["ProviderRegistry"] = None,
        provider_credentials: dict[str, ProviderCredentials] | None = None,
    ):
        """
        Args:
            registry_path: models.yaml 的路径。None 时使用项目默认路径
                           (configs/models.yaml，相对项目根目录)
            provider_registry: ProviderRegistry 实例，优先使用（支持运行时 CRUD）。
                           若提供，_resolve_credentials 将动态查询，修改立即生效。
            provider_credentials: 静态 provider 凭证字典（兜底，主要供测试使用）。
                           provider_registry 存在时此参数被忽略。
        """
        self._path = registry_path or self._default_path()
        # 优先使用 ProviderRegistry（支持运行时更新）；
        # 兜底使用静态字典（兼容旧调用路径和单元测试）
        self._provider_registry: ProviderRegistry | None = provider_registry
        self._provider_credentials: dict[str, ProviderCredentials] = (
            {}
            if provider_registry is not None
            else {name.lower(): cred for name, cred in (provider_credentials or {}).items()}
        )
        # 按插入顺序保存，以 id 为键
        self._models: dict[str, ModelDefinition] = {}
        self._load()

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------

    @staticmethod
    def _default_path() -> Path:
        """解析默认注册表文件路径（configs/models.yaml）"""
        # model_registry.py 位于 src/hivememory/system/，向上三级到达项目根目录
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "configs" / "models.yaml"

    # ------------------------------------------------------------------
    # 加载 & 持久化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """从 YAML 文件加载模型列表。文件不存在时以空注册表启动。"""
        if not self._path.exists():
            logger.warning("模型注册表文件未找到: %s，将以空注册表运行", self._path)
            return

        try:
            with open(self._path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_models: list[dict] = data.get("models", [])
            for raw in raw_models:
                # 残留的 api_key、api_base 不静默忽略：忽略它们会让请求改用 provider
                # 的密钥或地址，而不是写入者以为的那一个。
                if any(raw.get(field) not in (None, "") for field in ("api_key", "api_base")):
                    logger.error(
                        "跳过模型定义 %s：模型定义不保存 api_key 与 api_base，"
                        "它们属于 Provider 凭证；删除这两个字段，写入 .env 的 "
                        "HIVEMEMORY__PROVIDERS__<名称>__API_KEY / __API_BASE 或在 Provider 设置中"
                        "保存，需要不同的密钥或地址时另登记一个 Provider",
                        raw.get("id", "?"),
                    )
                    continue
                try:
                    model = ModelDefinition(**raw)
                    self._models[model.id] = model
                except Exception as e:
                    logger.error("跳过无效模型定义 %s: %s", raw.get("id", "?"), e)

            logger.info("已加载 %s 个模型定义（来自 %s）", len(self._models), self._path)

        except Exception as e:
            logger.error("加载模型注册表失败: %s", e)

    def _save(self) -> None:
        """
        将内存中的模型列表原子性地写入 YAML 文件。
        先写临时文件，再用 os.replace() 原子替换，避免写入过程中文件损坏。
        """
        payload = {"models": [model.model_dump(mode="json") for model in self._models.values()]}

        # 写入同目录的临时文件，再原子替换
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent,
            suffix=".tmp",
            prefix="models_",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    payload,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            os.replace(tmp_path, self._path)
            logger.debug("模型注册表已持久化（%s 条记录）", len(self._models))
        except Exception:
            # 清理临时文件后重新抛出
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelDefinition]:
        """返回所有模型定义，顺序与 YAML 文件一致。"""
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelDefinition | None:
        """
        按 ID 获取模型定义。

        Returns:
            找到返回 ModelDefinition，否则返回 None
        """
        return self._models.get(model_id)

    def get_default_model(self) -> ModelDefinition | None:
        """
        获取默认模型（is_default=True 的那条）。

        若没有任何模型标记为默认，退化返回第一条记录（如果存在）。
        """
        for model in self._models.values():
            if model.is_default:
                return model
        # 兜底：返回第一条
        return next(iter(self._models.values()), None)

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------

    def add_model(self, model: ModelDefinition) -> None:
        """
        添加新模型到注册表。

        若新模型 is_default=True，会自动将其他模型的 is_default 置为 False，
        确保注册表中始终最多只有一个默认模型。

        Raises:
            DuplicateModelIdError: 同 ID 模型已存在
        """
        if model.id in self._models:
            raise DuplicateModelIdError(f"模型 ID '{model.id}' 已存在")

        if model.is_default:
            self._clear_default_flag()

        self._models[model.id] = model
        self._save()
        logger.info("已添加模型: %s (%s)", model.id, model.display_name)

    def update_model(self, model_id: str, updates: dict) -> ModelDefinition:
        """
        更新指定模型的字段。

        Args:
            model_id: 要更新的模型 ID
            updates: 需要更新的字段字典（只含需要修改的字段）

        Returns:
            更新后的 ModelDefinition

        Raises:
            ModelNotFoundError: 模型不存在
        """
        existing = self._models.get(model_id)
        if existing is None:
            raise ModelNotFoundError(f"模型 '{model_id}' 不存在")

        # 合并更新
        merged = existing.model_dump()
        merged.update(updates)
        updated = ModelDefinition(**merged)

        # 若新数据将此模型设为默认，先清除其他模型的默认标记
        if updated.is_default and not existing.is_default:
            self._clear_default_flag()

        self._models[model_id] = updated
        self._save()
        logger.info("已更新模型: %s", model_id)
        return updated

    def delete_model(self, model_id: str) -> None:
        """
        从注册表中删除指定模型。

        Raises:
            ModelNotFoundError: 模型不存在
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"模型 '{model_id}' 不存在")

        del self._models[model_id]
        self._save()
        logger.info("已删除模型: %s", model_id)

    # ------------------------------------------------------------------
    # 模型解析
    # ------------------------------------------------------------------

    def _resolve_credentials(self, model: ModelDefinition) -> tuple[str | None, str | None]:
        """解析模型的 (api_key, api_base)。

        两者只来自 provider 凭证：models.yaml 受版本控制，模型定义不保存连接凭证。
        没有对应的 provider 凭证时为 None，由 litellm 从环境变量读取。

        provider 凭证查询：
        - 若注入了 ProviderRegistry，动态查询（支持运行时 CRUD，修改立即生效）
        - 否则使用构造时传入的静态字典（兼容测试 / 旧调用路径）
        """
        # 查询 provider 凭证（动态注册表优先，兜底静态字典）
        cred = None
        if model.provider:
            if self._provider_registry is not None:
                cred = self._provider_registry.get(model.provider)
            else:
                cred = self._provider_credentials.get(model.provider.lower())

        if cred is None:
            return None, None
        return cred.api_key, cred.api_base

    def select_model(self, model_id: str) -> ModelDefinition:
        """按 ID 选出模型定义；'default' 选出注册表的默认模型。

        Raises:
            ModelNotFoundError: 模型不存在，或注册表为空时选 'default'
        """
        if model_id == "default":
            model = self.get_default_model()
            if model is None:
                raise ModelNotFoundError("注册表为空，无法解析默认模型")
            return model
        model = self._models.get(model_id)
        if model is None:
            raise ModelNotFoundError(f"模型 '{model_id}' 不存在于注册表中")
        return model

    def resolve(self, selection: ModelSelection) -> LLMRequestParams:
        """把使用方的选择解析为这次 LLM 调用的请求参数。

        Alice 与 Gateway、Patchouli 的组件共用这一入口：模型取选中定义的
        litellm_model（``model_id`` 为 None 时取默认模型），凭证取它的 provider 凭证；
        生成参数以选择中的覆盖值优先，为 None 时取模型定义的默认值。

        Raises:
            ModelNotFoundError: 同 select_model
        """
        model = self.select_model(selection.model_id or "default")
        api_key, api_base = self._resolve_credentials(model)
        return LLMRequestParams(
            model=model.litellm_model,
            api_key=api_key,
            api_base=api_base,
            temperature=_or_default(selection.temperature, model.temperature),
            top_p=_or_default(selection.top_p, model.top_p),
            max_tokens=_or_default(selection.max_tokens, model.max_tokens),
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _clear_default_flag(self) -> None:
        """将所有模型的 is_default 设为 False。"""
        updated: dict[str, ModelDefinition] = {}
        for mid, m in self._models.items():
            if m.is_default:
                updated[mid] = m.model_copy(update={"is_default": False})
            else:
                updated[mid] = m
        self._models = updated

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"ModelRegistry(models={list(self._models.keys())}, path={self._path})"


def _or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value
