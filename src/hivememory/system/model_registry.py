"""
HiveMemory 模型注册表服务

ModelRegistry 是管理所有可用 LLM 模型的单一数据源（SSOT）。
持久化存储在 configs/models.yaml，支持运行时 CRUD 操作。

使用方式：
    registry = ModelRegistry()
    llm_config, display_name = registry.resolve("deepseek-chat")
    # 或者 model_name='default' 自动选取默认模型
    llm_config, display_name = registry.resolve("default")
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from hivememory.core.models.model_definition import ModelDefinition
from hivememory.system.config.shared import LLMConfig, ProviderCredentials

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """注册表中找不到指定 ID 的模型"""


class DuplicateModelIdError(Exception):
    """尝试添加已存在的模型 ID"""


class ModelRegistry:
    """
    模型注册表 — 管理系统中所有可用的 LLM 模型

    职责：
    - 从 configs/models.yaml 加载模型列表
    - 提供 CRUD 操作（list / get / add / update / delete）
    - 将 ModelDefinition 转换为运行时所需的 LLMConfig
    - 将变更原子性地持久化到 YAML 文件

    不变量：
    - 注册表中有且仅有一条记录的 is_default=True
    - 若注册表为空或没有默认模型，resolve("default") 返回列表中第一条记录
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        provider_credentials: Optional[Dict[str, ProviderCredentials]] = None,
    ):
        """
        Args:
            registry_path: models.yaml 的路径。None 时使用项目默认路径
                           (configs/models.yaml，相对项目根目录)
            provider_credentials: provider 名 → 凭证 的映射（来自 SharedConfig.providers）。
                           模型自身未显式设置 api_key/api_base 时，按 provider 补齐。
        """
        self._path = registry_path or self._default_path()
        # provider 名统一小写，与 ModelDefinition.provider 匹配
        self._provider_credentials: Dict[str, ProviderCredentials] = {
            name.lower(): cred for name, cred in (provider_credentials or {}).items()
        }
        # 按插入顺序保存，以 id 为键
        self._models: Dict[str, ModelDefinition] = {}
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
            logger.warning(f"模型注册表文件未找到: {self._path}，将以空注册表运行")
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_models: List[dict] = data.get("models", [])
            for raw in raw_models:
                try:
                    model = ModelDefinition(**raw)
                    self._models[model.id] = model
                except Exception as e:
                    logger.error(f"跳过无效模型定义 {raw.get('id', '?')}: {e}")

            logger.info(f"已加载 {len(self._models)} 个模型定义（来自 {self._path}）")

        except Exception as e:
            logger.error(f"加载模型注册表失败: {e}")

    def _save(self) -> None:
        """
        将内存中的模型列表原子性地写入 YAML 文件。
        先写临时文件，再用 os.replace() 原子替换，避免写入过程中文件损坏。
        """
        payload = {
            "models": [
                model.model_dump(mode="json")
                for model in self._models.values()
            ]
        }

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
            logger.debug(f"模型注册表已持久化（{len(self._models)} 条记录）")
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

    def list_models(self) -> List[ModelDefinition]:
        """返回所有模型定义，顺序与 YAML 文件一致。"""
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        """
        按 ID 获取模型定义。

        Returns:
            找到返回 ModelDefinition，否则返回 None
        """
        return self._models.get(model_id)

    def get_default_model(self) -> Optional[ModelDefinition]:
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
        logger.info(f"已添加模型: {model.id} ({model.display_name})")

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
        logger.info(f"已更新模型: {model_id}")
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
        logger.info(f"已删除模型: {model_id}")

    # ------------------------------------------------------------------
    # LLM 配置解析
    # ------------------------------------------------------------------

    def _resolve_credentials(
        self, model: ModelDefinition
    ) -> Tuple[Optional[str], Optional[str]]:
        """解析模型的 (api_key, api_base)。

        优先级：模型自身显式设置 > provider 凭证表 > None（litellm 环境变量兜底）。
        models.yaml 被 git 跟踪，通常模型自身留空，凭证由 provider 表（来自 .env）补齐。
        """
        cred = self._provider_credentials.get(model.provider.lower()) if model.provider else None
        api_key = model.api_key if model.api_key is not None else (cred.api_key if cred else None)
        api_base = model.api_base if model.api_base is not None else (cred.api_base if cred else None)
        return api_key, api_base

    def to_llm_config(self, model_id: str) -> LLMConfig:
        """
        将注册表中的模型定义转换为 LLMConfig。

        Args:
            model_id: 模型 ID

        Returns:
            LLMConfig 实例，可直接传给 LiteLLMService

        Raises:
            ModelNotFoundError: 模型不存在
        """
        model = self._models.get(model_id)
        if model is None:
            raise ModelNotFoundError(f"模型 '{model_id}' 不存在")

        api_key, api_base = self._resolve_credentials(model)
        return LLMConfig(
            model=model.litellm_model,
            api_key=api_key,
            api_base=api_base,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
        )

    def resolve_for_llm_config(self, llm_config: "LLMConfig") -> "LLMConfig":
        """
        将带有 model_id 的 LLMConfig（如 gateway/librarian 配置）解析为可直接使用的 LLMConfig。

        规则：
        - llm_config.model_id 有值 → 通过注册表查 litellm_model + 补齐凭证
        - llm_config.model_id 为空 → 直接使用 llm_config.model（向后兼容旧配置）
        - temperature / max_tokens / top_p 始终以 llm_config 自身值为准（组件调参优先于注册表默认）

        Args:
            llm_config: 含有 model_id（或旧 model 字符串）的配置

        Returns:
            完全填充（model + api_key + api_base + 温度参数）的 LLMConfig

        Raises:
            ModelNotFoundError: model_id 不在注册表中，且注册表非空时
        """
        if not llm_config.model_id:
            # 向后兼容路径：没有 model_id，直接使用 model 字段（旧式 litellm 字符串）
            return llm_config

        # 解析注册表模型
        if llm_config.model_id == "default":
            model = self.get_default_model()
            if model is None:
                raise ModelNotFoundError("注册表为空，无法解析默认模型")
        else:
            model = self._models.get(llm_config.model_id)
            if model is None:
                raise ModelNotFoundError(
                    f"模型 '{llm_config.model_id}' 不存在于注册表中"
                )

        api_key, api_base = self._resolve_credentials(model)
        return LLMConfig(
            # 模型名和凭证来自注册表
            model=model.litellm_model,
            api_key=api_key,
            api_base=api_base,
            # temperature / max_tokens / top_p 以组件自身配置为准（覆盖注册表默认值）
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            top_p=llm_config.top_p,
        )

    def resolve(
        self,
        model_name: str,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        top_p_override: Optional[float] = None,
    ) -> Tuple[LLMConfig, str]:
        """
        解析模型名称，返回运行时所需的 LLMConfig 和展示名称。

        这是 Agent 运行时使用的核心方法，实现了配置优先级链：
            会话/Profile 覆盖（temperature/max_tokens/top_p） > 模型注册表默认值

        Args:
            model_name: 模型 ID，或 'default' 表示使用注册表默认模型
            temperature_override: 温度覆盖，None 表示使用模型默认值
            max_tokens_override: max_tokens 覆盖，None 表示使用模型默认值
            top_p_override: top_p 覆盖，None 表示使用模型默认值

        Returns:
            (LLMConfig, display_name) — 配置对象和前端可展示的模型名称

        Raises:
            ModelNotFoundError: model_name 不是 'default' 且对应 ID 不存在
        """
        if model_name == "default":
            model = self.get_default_model()
            if model is None:
                raise ModelNotFoundError("注册表为空，无法解析默认模型")
        else:
            model = self._models.get(model_name)
            if model is None:
                raise ModelNotFoundError(f"模型 '{model_name}' 不存在于注册表中")

        api_key, api_base = self._resolve_credentials(model)
        config = LLMConfig(
            model=model.litellm_model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature_override if temperature_override is not None else model.temperature,
            max_tokens=max_tokens_override if max_tokens_override is not None else model.max_tokens,
            top_p=top_p_override if top_p_override is not None else model.top_p,
        )
        return config, model.display_name

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _clear_default_flag(self) -> None:
        """将所有模型的 is_default 设为 False。"""
        updated: Dict[str, ModelDefinition] = {}
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
