import json
from typing import Any, Dict, List

from app.agents.critic.prompts import CRITIC_SYSTEM_PROMPT
from app.agents.critic.schemas import CriticResult
from app.llm.yandex_llm_client import YandexLLMClient
from app.domain.agent_response import CarRecommendation
from app.domain.car import Car
from app.domain.user_session import UserSession


class CriticAgent:
    """Агент проверки качества результата."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
    ) -> None:
        self.llm_client = llm_client

    def review(
            self,
            session: UserSession,
            recommendations: List[CarRecommendation],
            tool_cars: List[Car],
            user_message: str,
    ) -> CriticResult:
        """Проверяет рекомендации кодом и через LLM."""

        code_issues = self._run_code_checks(
            session=session,
            recommendations=recommendations,
            tool_cars=tool_cars,
        )

        if code_issues:
            return CriticResult(
                approved=False,
                issues=code_issues,
                fixed_user_message="",
            )

        llm_result = self._run_llm_checks(
            session=session,
            recommendations=recommendations,
            tool_cars=tool_cars,
            user_message=user_message,
        )

        return llm_result

    def _run_code_checks(
            self,
            session: UserSession,
            recommendations: List[CarRecommendation],
            tool_cars: List[Car],
    ) -> List[str]:
        """Проверяет формальные правила без LLM."""

        issues: List[str] = []

        if not recommendations:
            issues.append("Список рекомендаций пустой.")

        if len(recommendations) > 5:
            issues.append("Агент рекомендовал больше 5 машин.")

        tool_cars_by_id = {car.id: car for car in tool_cars}

        for recommendation in recommendations:
            car = tool_cars_by_id.get(recommendation.car_id)

            if car is None:
                issues.append(f"Машина {recommendation.car_id} не была получена из tools.")
                continue

            if not car.fits_budget(
                    budget_min=session.budget_min,
                    budget_max=session.budget_max,
            ):
                issues.append(f"Машина {car.title()} не попадает в бюджет пользователя.")

            self._check_must_not_have(
                car=car,
                must_not_have=session.must_not_have,
                issues=issues,
            )

        return issues

    def _check_must_not_have(
            self,
            car: Car,
            must_not_have: List[str],
            issues: List[str],
    ) -> None:
        """Проверяет явные запреты пользователя."""

        car_text = " ".join(
            [
                car.brand,
                car.model,
                car.body_type.value,
                car.fuel_type.value,
                car.transmission.value,
                car.drive_type.value,
                car.description,
            ]
        ).lower()

        for forbidden_item in must_not_have:
            if forbidden_item.lower() in car_text:
                issues.append(
                    f"Машина {car.title()} нарушает запрет пользователя: {forbidden_item}."
                )

    def _run_llm_checks(
            self,
            session: UserSession,
            recommendations: List[CarRecommendation],
            tool_cars: List[Car],
            user_message: str,
    ) -> CriticResult:
        """Проверяет объяснения через LLM."""

        payload = {
            "user_session": self._session_to_dict(session),
            "recommended_cars": [
                {
                    "car_id": str(item.car_id),
                    "reason": item.reason,
                    "risk_note": item.risk_note,
                }
                for item in recommendations
            ],
            "tool_cars": [self._car_to_dict(car) for car in tool_cars],
            "user_message": user_message,
        }

        raw_response = self.llm_client.generate(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=CriticResult.model_json_schema(),
            response_schema_name="critic_result",
        )

        return CriticResult.model_validate_json(raw_response)

    def _session_to_dict(self, session: UserSession) -> Dict[str, Any]:
        """Преобразует UserSession в словарь для проверки."""

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

    def _car_to_dict(self, car: Car) -> Dict[str, Any]:
        """Преобразует Car в словарь для проверки."""

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
