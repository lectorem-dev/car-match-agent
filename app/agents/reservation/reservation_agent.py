import json
from typing import Any, Dict, Optional

from app.agents.reservation.reservation_prompts import RESERVATION_SYSTEM_PROMPT
from app.agents.reservation.reservation_schemas import (
    ReservationIntent,
    ReservationResult,
)
from app.agent_tools.car_catalog import CarCatalog
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class ReservationAgent:
    """Агент обработки mock-заявки на бронь."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            catalog: CarCatalog,
            enable_logs: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.catalog = catalog
        self.logger = AgentLogger(
            "ReservationAgent",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_GREEN,
        )

    def handle(
            self,
            user_message: str,
            session: UserSession,
            scenario_name: Optional[str] = None,
    ) -> ReservationResult:
        """Обрабатывает запрос на бронь."""

        self.logger.start(
            scenario=scenario_name,
            user_message=user_message,
            dialog_status=session.dialog_status.value,
            selected_car_id=str(session.selected_car_id) if session.selected_car_id else None,
        )

        try:
            result = self._make_decision(
                user_message=user_message,
                session=session,
            )

            if result.intent != ReservationIntent.NOT_RESERVATION_REQUEST:
                self._try_select_car_by_title(
                    session=session,
                    selected_car_title=result.selected_car_title,
                )

            self.logger.success(
                scenario=scenario_name,
                intent=result.intent.value,
                selected_car_title=result.selected_car_title,
                current_selected_car_id=str(session.selected_car_id) if session.selected_car_id else None,
            )
            return result
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                user_message=user_message,
                has_current_session=session is not None,
                current_selected_car_id=str(session.selected_car_id) if session and session.selected_car_id else None,
                none_object=detect_none_object_name(
                    error,
                    current_session=session,
                ),
            )
            raise

    def is_reservation_request(
            self,
            user_message: str,
            session: UserSession,
    ) -> bool:
        """Проверяет, является ли сообщение просьбой о бронировании."""

        decision = self._make_decision(
            user_message=user_message,
            session=session,
        )

        return decision.intent == ReservationIntent.RESERVATION_REQUEST

    def _make_decision(
            self,
            user_message: str,
            session: UserSession,
    ) -> ReservationResult:
        """Определяет намерение бронирования."""

        payload = {
            "user_message": user_message,
            "current_session": self._session_to_dict(session),
        }

        raw_response = self.llm_client.generate(
            system_prompt=RESERVATION_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=ReservationResult.model_json_schema(),
            response_schema_name="reservation_result",
        )

        return ReservationResult.model_validate_json(raw_response)

    def _try_select_car_by_title(
            self,
            session: UserSession,
            selected_car_title: Optional[str],
    ) -> None:
        """Пытается выбрать машину по названию из сообщения пользователя."""

        if not selected_car_title:
            return

        selected_car_title_lower = selected_car_title.lower()

        for car in self.catalog.find_all():
            if car.title().lower() == selected_car_title_lower:
                session.select_car(car.id)
                return

            short_title = f"{car.brand} {car.model}".lower()
            if short_title == selected_car_title_lower:
                session.select_car(car.id)
                return

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
