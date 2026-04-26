import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from dotenv import load_dotenv

YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
YANDEX_MODEL_NAME = "gpt-oss-20b/latest"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 700

# Цвет логов LLM-клиента.
LLM_LOG_COLOR = "\033[92m"
LLM_LOG_RESET = "\033[0m"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("car_match_agent.llm")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[LLM] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


LOGGER = _build_logger()


class YandexLLMClient:
    """Клиент для вызова Yandex AI Studio через OpenAI-совместимый API."""

    def __init__(
            self,
            folder_id: Optional[str] = None,
            api_key: Optional[str] = None,
            model_name: str = YANDEX_MODEL_NAME,
            base_url: str = YANDEX_BASE_URL,
            enable_logging: bool = True,
    ) -> None:
        # .env лежит в корне проекта.
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)

        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self.enable_logging = enable_logging

        if not self.folder_id:
            raise ValueError("Не найден YANDEX_FOLDER_ID в .env")

        if not self.api_key:
            raise ValueError("Не найден YANDEX_API_KEY в .env")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            project=self.folder_id,
        )

        self._log("LLM клиент инициализирован")
        self._log("model=%s", self.model_name)
        self._log("base_url=%s", self.base_url)

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = DEFAULT_TEMPERATURE,
            max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
            response_schema: Optional[Dict[str, Any]] = None,
            response_schema_name: str = "structured_response",
    ) -> str:
        """Вызывает модель и возвращает текст ответа."""

        model = self._full_model_name()

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
        }

        # Structured output включается только если передали JSON Schema.
        if response_schema is not None:
            request_kwargs["text"] = self._build_json_schema_format(
                schema=response_schema,
                schema_name=response_schema_name,
            )

        self._log("Вызов модели начат")
        self._log("model=%s", model)
        self._log("temperature=%s", temperature)
        self._log("max_output_tokens=%s", max_output_tokens)
        self._log("structured_output=%s", response_schema is not None)
        self._log("system_prompt=%s", self._shorten(system_prompt, 300))
        self._log("user_prompt=%s", self._shorten(user_prompt, 500))

        response = self.client.responses.create(**request_kwargs)

        output_text = (response.output_text or "").strip()

        self._log("Вызов модели завершен")
        self._log("response_length=%s", len(output_text))
        self._log("response=%s", self._shorten(output_text, 500))
        self._log("response_repr=%r", output_text)

        if not output_text:
            raise ValueError("Модель вернула пустой ответ")

        return output_text

    @staticmethod
    def _build_json_schema_format(
            schema: Dict[str, Any],
            schema_name: str,
    ) -> Dict[str, Any]:
        """Собирает формат structured output для Responses API."""
        return {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        }

    def _full_model_name(self) -> str:
        """Собирает полный путь до модели Yandex."""
        return "gpt://{folder}/{model}".format(
            folder=self.folder_id,
            model=self.model_name,
        )

    def _log(self, message: str, *args) -> None:
        """Пишет лог, если логирование включено."""
        if not self.enable_logging:
            return
        LOGGER.info(f"{LLM_LOG_COLOR}{message}{LLM_LOG_RESET}", *args)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        """Обрезает длинный текст для логов."""
        clean = " ".join((text or "").split())

        if len(clean) <= limit:
            return clean

        return clean[:limit] + "..."