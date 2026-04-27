import json
from typing import Any, Dict, Optional

from app.agents.guard.guard_prompts import DOMAIN_GUARD_SYSTEM_PROMPT
from app.agents.guard.guard_schemas import DomainGuardResult, DomainStatus
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class DomainGuardAgent:
    """Агент проверки предметной области запроса."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            enable_logs: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.logger = AgentLogger(
            "DomainGuardAgent",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_BLUE,
        )

    def check(
            self,
            user_message: str,
            session: UserSession,
            scenario_name: Optional[str] = None,
    ) -> DomainGuardResult:
        """Проверяет, относится ли запрос к автомобильному домену."""

        self.logger.start(
            scenario=scenario_name,
            user_message=user_message,
            dialog_status=session.dialog_status.value,
        )

        try:
            payload = {
                "user_message": user_message,
                "current_session": self._session_to_dict(session),
            }

            raw_response = self.llm_client.generate(
                system_prompt=DOMAIN_GUARD_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                response_schema=DomainGuardResult.model_json_schema(),
                response_schema_name="domain_guard_result",
            )

            result = DomainGuardResult.model_validate_json(raw_response)
            self.logger.success(
                scenario=scenario_name,
                domain_status=result.domain_status.value,
            )
            return result
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                user_message=user_message,
                has_current_session=session is not None,
                none_object=detect_none_object_name(
                    error,
                    current_session=session,
                ),
            )
            raise

    def is_in_domain(
            self,
            user_message: str,
            session: UserSession,
            scenario_name: Optional[str] = None,
    ) -> bool:
        """Быстрая проверка: запрос в домене или нет."""

        result = self.check(
            user_message=user_message,
            session=session,
            scenario_name=scenario_name,
        )

        return result.domain_status == DomainStatus.IN_DOMAIN

    @staticmethod
    def _session_to_dict(session: UserSession) -> Dict[str, Any]:
        """Преобразует UserSession в словарь для промпта."""

        return {
            "budget_min": session.budget_min,
            "budget_max": session.budget_max,
            "purpose": session.purpose,
            "experience_level": session.experience_level,
            "family_size": session.family_size,
            "preferred_body_types": session.preferred_body_types,
            "preferred_brands": session.preferred_brands,
            "must_have": session.must_have,
            "must_not_have": session.must_not_have,
            "user_notes": session.user_notes,
            "selected_car_id": str(session.selected_car_id) if session.selected_car_id else None,
            "dialog_status": session.dialog_status.value,
        }
