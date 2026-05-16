"""Syscall live 场景测试共享组件。"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from hivememory.system.config import LLMConfig, KoakumaConfig
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.core.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
)
from hivememory.prompts.mtp import MTPPromptBuilder

logger = logging.getLogger(__name__)


class MTPLoopRunner:
    def __init__(
        self,
        llm_service,
        koakuma: KoakumaRuntime,
        max_rounds: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.llm_service = llm_service
        self.koakuma = koakuma
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.round_log: List[Dict] = []

    @staticmethod
    def _log_separator(title: str, char: str = "=", width: int = 72):
        logger.info(f"\n{char * width}")
        logger.info(f"  {title}")
        logger.info(f"{char * width}")

    @staticmethod
    def _log_messages_summary(messages: List[Dict[str, str]]):
        logger.info(f"  Messages stack ({len(messages)} items):")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            if role == "system":
                logger.info(f"    [{i}] system  | ({len(content)} chars, prompt omitted)")
            else:
                preview = content.replace("\n", "\\n")[:120]
                logger.info(f"    [{i}] {role:9s} | {preview}...")

    def run(self, system_prompt: str, user_message: str) -> Tuple[str, List[Dict[str, str]]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        accumulated_text = ""
        self.round_log = []
        self._log_separator(f'MTP LOOP START | user: "{user_message[:60]}"', "━")

        for round_idx in range(self.max_rounds):
            self._log_separator(f"Round {round_idx + 1}/{self.max_rounds}", "─")
            response_text = self.llm_service.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=[MTP_STOP_SEQUENCE],
            )
            accumulated_text += response_text
            round_info = {
                "round": round_idx + 1,
                "llm_output": response_text,
                "mtp_triggered": False,
                "mtp_result": None,
            }

            if MTP_LEFT_DELIMITER in response_text:
                round_info["mtp_triggered"] = True
                result = self.koakuma.intercept_and_execute(accumulated_text)
                if result is not None and result.formatted_response:
                    round_info["mtp_result"] = {
                        "success": result.success,
                        "status": result.response_status,
                        "content_preview": result.response_content[:200],
                    }
                    backfill_text = (
                        accumulated_text + MTP_RIGHT_DELIMITER
                        + "\n" + result.formatted_response.split("\n", 1)[-1]
                    )
                    messages.append({"role": "assistant", "content": backfill_text})
                    accumulated_text = ""
                    self.round_log.append(round_info)
                    continue

            self.round_log.append(round_info)
            break

        return accumulated_text, messages


def _get_llm_config() -> Optional[LLMConfig]:
    model = os.environ.get("MTP_TEST_MODEL")
    api_key = os.environ.get("MTP_TEST_API_KEY")
    if model and api_key:
        return LLMConfig(
            model=model,
            api_key=api_key,
            api_base=os.environ.get("MTP_TEST_API_BASE"),
            temperature=0.0,
            max_tokens=1024,
        )
    try:
        from hivememory.system.config import load_app_config

        config = load_app_config()
        llm_config = config.get_librarian_llm_config()
        if llm_config and llm_config.model:
            return llm_config
    except Exception:
        pass
    return None


def _create_llm_service(config: LLMConfig):
    class TestLLMService:
        def __init__(self, cfg: LLMConfig):
            self.model = cfg.model
            self.api_key = cfg.api_key
            self.api_base = cfg.api_base
            self.temperature = cfg.temperature
            self.max_tokens = cfg.max_tokens

        def complete(self, messages, temperature=None, max_tokens=None, **kwargs) -> str:
            import litellm

            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content

    return TestLLMService(config)


def _create_koakuma() -> KoakumaRuntime:
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(),
    )


def _build_mtp_system_prompt(language: str = "en") -> str:
    base_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's questions accurately and concisely."
    )
    mtp_fragment = MTPPromptBuilder(
        language=language,
        kernel_tools=[
            ("sys_clock", "Get current date, time, and timezone."),
            ("sys_python_repl", "Execute Python code for calculation or data processing."),
        ],
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


def _build_mtp_system_prompt_with_file_io(language: str = "en") -> str:
    base_prompt = (
        "You are a helpful AI assistant with access to a workspace directory. "
        "You can read and write files in the workspace using MTP tools. "
        "Answer the user's questions accurately and concisely."
    )
    mtp_fragment = MTPPromptBuilder(
        language=language,
        kernel_tools=[
            ("sys_clock", "Get current date, time, and timezone."),
            ("sys_python_repl", "Execute Python code for calculation or data processing."),
            ("sys_read_file", "Read a file from the workspace directory. Args: path (relative path)."),
            (
                "sys_write_file",
                "Write content to a file in the workspace directory. Args: path (relative path), content (text to write), mode (overwrite|append, default overwrite).",
            ),
        ],
    ).build()
    return f"{base_prompt}\n\n{mtp_fragment}"


