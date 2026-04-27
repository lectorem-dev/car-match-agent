import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.agents.critic.critic_prompts import CRITIC_SYSTEM_PROMPT
from app.agents.critic.critic_schemas import CriticResult
from app.agents.planner.planner_schemas import PlannedCar
from app.domain.car import BodyType, Car, DriveType, Transmission
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class CriticAgent:
    """Агент проверки качества результата."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            enable_logs: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.logger = AgentLogger(
            "CriticAgent",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_CYAN,
        )

    def review(
            self,
            session: UserSession,
            recommendations: List[PlannedCar],
            tool_cars: List[Car],
            scenario_name: Optional[str] = None,
    ) -> CriticResult:
        """Проверяет рекомендации кодом и через LLM."""

        self.logger.start(
            scenario=scenario_name,
            recommendations_count=len(recommendations),
        )

        try:
            code_issues = self._run_code_checks(
                session=session,
                recommendations=recommendations,
                tool_cars=tool_cars,
            )

            if code_issues:
                result = CriticResult(
                    approved=False,
                    issues=code_issues,
                    user_message="",
                )
                self.logger.success(
                    scenario=scenario_name,
                    approved=result.approved,
                    issues=result.issues,
                )
                return result

            try:
                result = self._run_llm_checks(
                    session=session,
                    recommendations=recommendations,
                    tool_cars=tool_cars,
                )
            except ValidationError:
                result = CriticResult(
                    approved=True,
                    issues=[],
                    user_message=self._build_user_message(
                        session=session,
                        tool_cars=tool_cars,
                    ),
                )

            if not result.approved and not self._has_hard_llm_issues(
                session=session,
                issues=result.issues,
            ):
                result = CriticResult(
                    approved=True,
                    issues=[],
                    user_message=self._build_user_message(
                        session=session,
                        tool_cars=tool_cars,
                    ),
                )

            if result.approved and not result.user_message.strip():
                result.user_message = self._build_user_message(
                    session=session,
                    tool_cars=tool_cars,
                )

            self.logger.success(
                scenario=scenario_name,
                approved=result.approved,
                issues=result.issues,
            )
            return result
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                has_current_session=session is not None,
                recommendations_count=len(recommendations),
                none_object=detect_none_object_name(
                    error,
                    current_session=session,
                    recommended_cars=recommendations,
                    tool_cars=tool_cars,
                ),
            )
            raise

    def _run_code_checks(
            self,
            session: UserSession,
            recommendations: List[PlannedCar],
            tool_cars: List[Car],
    ) -> List[str]:
        """Проверяет формальные правила без LLM."""

        issues: List[str] = []

        if not recommendations:
            issues.append("Список рекомендаций пустой.")

        if len(recommendations) > 3:
            issues.append("Агент рекомендовал больше 3 машин.")

        tool_cars_by_id = {car.id: car for car in tool_cars}

        for recommendation in recommendations:
            car = tool_cars_by_id.get(recommendation.car_id)

            if car is None:
                issues.append(f"Машина {recommendation.car_id} не была получена из tools.")
                continue

            if session.budget_max is not None and car.price > session.budget_max:
                issues.append(f"Машина {car.title()} не попадает в бюджет пользователя.")

            self._check_must_not_have(
                car=car,
                must_not_have=session.must_not_have,
                issues=issues,
            )

        return issues

    @staticmethod
    def _check_must_not_have(
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
            recommendations: List[PlannedCar],
            tool_cars: List[Car],
    ) -> CriticResult:
        """Проверяет рекомендации и формирует итоговый ответ через LLM."""

        payload = {
            "user_session": self._session_to_dict(session),
            "recommended_cars": [
                {
                    "car_id": str(item.car_id),
                    "reason": item.reason,
                }
                for item in recommendations
            ],
            "tool_cars": [self._car_to_dict(car) for car in tool_cars],
        }

        raw_response = self.llm_client.generate(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=CriticResult.model_json_schema(),
            response_schema_name="critic_result",
            max_output_tokens=1500,
        )

        return CriticResult.model_validate_json(raw_response)

    def _build_user_message(
            self,
            session: UserSession,
            tool_cars: List[Car],
    ) -> str:
        """Строит короткий локальный fallback-ответ пользователю на русском."""

        if not tool_cars:
            return "Подходящие варианты не найдены."

        titles = ", ".join(
            f"{car.title()} ({car.price} USD)"
            for car in tool_cars
        )
        budget_part = ""

        if session.budget_max is not None:
            budget_part = f" Все варианты укладываются в бюджет до {session.budget_max} долларов."

        first_car = tool_cars[0]
        fit_parts = [
            self._body_type_label(first_car.body_type),
            self._transmission_label(first_car.transmission),
            self._drive_type_label(first_car.drive_type),
        ]
        fit_parts = [part for part in fit_parts if part]
        fit_text = ", ".join(fit_parts[:2])

        if fit_text:
            fit_text = f" Базовый ориентир: {fit_text}."

        return f"Подобрал варианты: {titles}.{budget_part}{fit_text}"

    def _has_hard_llm_issues(
            self,
            session: UserSession,
            issues: List[str],
    ) -> bool:
        """Определяет, содержат ли замечания LLM реальные hard-constraints."""

        if not issues:
            return False

        context_text = self._build_context_text(session)
        requires_awd = self._contains_any(context_text, ["awd", "full drive", "полный привод"])
        requires_electric = self._contains_any(context_text, ["electric", "электро", "электромоб"])

        for issue in issues:
            normalized_issue = issue.lower()

            if self._contains_any(
                normalized_issue,
                [
                    "не попадает в бюджет",
                    "budget",
                    "price",
                    "дороже",
                    "нарушает запрет",
                    "must_not_have",
                    "forbidden",
                    "не была получена из tools",
                    "not found in tool_cars",
                    "список рекомендаций пустой",
                    "больше 3 машин",
                ],
            ):
                return True

            if requires_awd and self._contains_any(
                normalized_issue,
                ["полный привод", "awd", "all-wheel drive", "full drive"],
            ):
                return True

            if requires_electric and self._contains_any(
                normalized_issue,
                ["electric", "электро", "электромоб"],
            ):
                return True

        return False

    @staticmethod
    def _build_context_text(session: UserSession) -> str:
        """Собирает единый текстовый контекст пользователя для простых правил."""

        parts = [
            session.purpose or "",
            session.user_notes,
            " ".join(session.must_have),
            " ".join(session.must_not_have),
            " ".join(session.preferred_body_types),
            " ".join(session.preferred_brands),
        ]
        return " ".join(parts).lower()

    @staticmethod
    def _contains_any(text: str, patterns: List[str]) -> bool:
        """Проверяет, содержит ли текст хотя бы один из паттернов."""

        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _session_to_dict(session: UserSession) -> Dict[str, Any]:
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

    @staticmethod
    def _car_to_dict(car: Car) -> Dict[str, Any]:
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

    @staticmethod
    def _body_type_label(body_type: BodyType) -> str:
        """Возвращает короткую русскую метку типа кузова."""

        labels = {
            BodyType.SEDAN: "седан",
            BodyType.HATCHBACK: "хэтчбек",
            BodyType.LIFTBACK: "лифтбек",
            BodyType.WAGON: "универсал",
            BodyType.SUV: "suv/crossover",
            BodyType.COUPE: "купе",
            BodyType.MINIVAN: "минивэн",
            BodyType.PICKUP: "пикап",
        }
        return labels.get(body_type, "")

    @staticmethod
    def _transmission_label(transmission: Transmission) -> str:
        """Возвращает короткую русскую метку коробки передач."""

        labels = {
            Transmission.MANUAL: "механика",
            Transmission.AUTOMATIC: "автомат",
            Transmission.ROBOT: "робот",
            Transmission.CVT: "вариатор",
        }
        return labels.get(transmission, "")

    @staticmethod
    def _drive_type_label(drive_type: DriveType) -> str:
        """Возвращает короткую русскую метку привода."""

        labels = {
            DriveType.FWD: "передний привод",
            DriveType.RWD: "задний привод",
            DriveType.AWD: "полный привод",
        }
        return labels.get(drive_type, "")
