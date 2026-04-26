import json
from typing import Any, Dict, List, Optional

from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.agents.planner.schemas import PlannerResult
from app.catalog.car_catalog import CarCatalog
from app.domain.car import Car
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class PlannerAgent:
    """Агент подбора автомобилей."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            catalog_tool: CarCatalog,
            enable_logs: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.catalog_tool = catalog_tool
        self.logger = AgentLogger(
            "PlannerAgent",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_MAGENTA,
        )

    def plan(
            self,
            session: UserSession,
            critic_issues: Optional[List[str]] = None,
            scenario_name: Optional[str] = None,
    ) -> PlannerResult:
        """Подбирает машины по текущей сессии пользователя."""

        candidates: Optional[List[Car]] = None

        try:
            candidates = self._find_candidates(session=session)

            self.logger.start(
                scenario=scenario_name,
                budget_max=session.budget_max,
                purpose=session.purpose,
                candidates_count=len(candidates),
            )

            if not candidates:
                raise ValueError("Не найдено машин-кандидатов для подбора.")

            user_prompt = self._build_user_prompt(
                session=session,
                candidates=candidates,
                critic_issues=critic_issues or [],
            )

            raw_response = self.llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=PlannerResult.model_json_schema(),
                response_schema_name="planner_result",
                max_output_tokens=1500,
            )

            result = PlannerResult.model_validate_json(raw_response)
            self.logger.success(
                scenario=scenario_name,
                recommendations_count=len(result.recommendations),
                car_ids=[str(item.car_id) for item in result.recommendations],
            )
            return result
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                has_user_session=session is not None,
                candidates_count=len(candidates) if candidates is not None else None,
                none_object=detect_none_object_name(
                    error,
                    user_session=session,
                    candidate_cars=candidates,
                ),
            )
            raise

    def _find_candidates(self, session: UserSession) -> List[Car]:
        """Находит кандидатов через каталог машин."""

        body_type = self._first_or_none(session.preferred_body_types)
        brand = self._first_or_none(session.preferred_brands)

        candidates = self.catalog_tool.find_by_filters(
            budget_min=session.budget_min,
            budget_max=session.budget_max,
            body_type=body_type,
            brand=brand,
        )

        # Если строгие фильтры дали мало вариантов, ослабляем поиск до бюджета.
        if len(candidates) < 3:
            candidates = self.catalog_tool.find_by_budget(
                budget_min=session.budget_min,
                budget_max=session.budget_max,
            )

        # Если бюджета нет, показываем весь каталог.
        if not candidates:
            candidates = self.catalog_tool.find_all()

        return candidates

    def _build_user_prompt(
            self,
            session: UserSession,
            candidates: List[Car],
            critic_issues: List[str],
    ) -> str:
        """Собирает вход для LLM."""

        payload = {
            "user_session": self._session_to_dict(session),
            "candidate_cars": [self._car_to_dict(car) for car in candidates],
            "critic_issues": critic_issues,
            "output_rule": "Выбери от 1 до 5 машин только из candidate_cars.",
        }

        return json.dumps(payload, ensure_ascii=False, indent=2)

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

    @staticmethod
    def _car_to_dict(car: Car) -> Dict[str, Any]:
        """Преобразует Car в словарь для промпта."""

        return {
            "id": str(car.id),
            "title": car.title(),
            "brand": car.brand,
            "model": car.model,
            "year": car.year,
            "price": car.price,
            "body_type": car.body_type.value,
            "mileage_km": car.mileage_km,
            "fuel_type": car.fuel_type.value,
            "transmission": car.transmission.value,
            "drive_type": car.drive_type.value,
            "engine_power_hp": car.engine_power_hp,
            "description": car.description,
        }

    @staticmethod
    def _first_or_none(values: List[str]) -> Optional[str]:
        """Возвращает первое значение списка или None."""
        if not values:
            return None

        return values[0]
