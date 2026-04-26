import json
from enum import Enum
from typing import Any, Dict, Optional


class AgentLogColor(str, Enum):
    """Базовые ANSI-цвета, хорошо читаемые на черном фоне."""

    BRIGHT_BLUE = "\033[94m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_WHITE = "\033[97m"
    RESET = "\033[0m"


def detect_none_object_name(error: Exception, **objects: Any) -> Optional[str]:
    """Пытается назвать объект, который оказался None при NoneType-ошибке."""

    if "NoneType" not in str(error):
        return None

    for name, value in objects.items():
        if value is None:
            return name

    return None


class AgentLogger:
    """Минимальный осмысленный logger для агентного пайплайна.

    Формат:
    [AgentName] event key=value key=value

    Принцип:
    - не печатает огромные prompt/response целиком;
    - показывает только то, что помогает понять путь выполнения;
    - умеет кратко резать длинные значения;
    - все агенты используют единый формат.
    """

    def __init__(
            self,
            agent_name: str,
            enabled: bool = True,
            max_value_len: int = 180,
            color: AgentLogColor = AgentLogColor.BRIGHT_WHITE,
    ):
        self.agent_name = agent_name
        self.enabled = enabled
        self.max_value_len = max_value_len
        self.color = color

    def event(self, event_name: str, **data: Any) -> None:
        if not self.enabled:
            return

        message = self._format_message(event_name, data)
        print(f"{self.color.value}{message}{AgentLogColor.RESET.value}")

    def start(self, **data: Any) -> None:
        self.event("start", **data)

    def success(self, **data: Any) -> None:
        self.event("success", **data)

    def fail(self, error: Exception, **data: Any) -> None:
        self.event(
            "fail",
            error_type=error.__class__.__name__,
            error=str(error),
            **data,
        )

    def llm_call(
            self,
            response_model: Optional[str] = None,
            max_output_tokens: Optional[int] = None,
            **data: Any,
    ) -> None:
        self.event(
            "llm_call",
            response_model=response_model,
            max_output_tokens=max_output_tokens,
            **data,
        )

    def llm_result(
            self,
            response_len: Optional[int] = None,
            parsed: Optional[bool] = None,
            **data: Any,
    ) -> None:
        self.event(
            "llm_result",
            response_len=response_len,
            parsed=parsed,
            **data,
        )

    def state(self, **data: Any) -> None:
        self.event("state", **data)

    def decision(self, decision: str, **data: Any) -> None:
        self.event("decision", decision=decision, **data)

    def _format_message(self, event_name: str, data: Dict[str, Any]) -> str:
        parts = [f"[{self.agent_name}]", event_name]

        for key, value in data.items():
            if value is None:
                continue

            parts.append(f"{key}={self._format_value(value)}")

        return " ".join(parts)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            text = value.replace("\n", " ").strip()
        elif isinstance(value, (int, float, bool)):
            text = str(value)
        elif isinstance(value, (list, tuple, set)):
            text = json.dumps(list(value), ensure_ascii=False)
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        if len(text) > self.max_value_len:
            text = text[: self.max_value_len] + "..."

        if " " in text:
            return repr(text)

        return text
