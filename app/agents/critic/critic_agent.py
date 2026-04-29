import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import ValidationError

from app.agents.critic.critic_prompts import CRITIC_SYSTEM_PROMPT
from app.agents.critic.critic_schemas import CriticCarReview, CriticResult
from app.agents.planner.planner_schemas import PlannedCar
from app.domain.car import BodyType, Car, DriveType, Transmission
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class CriticAgent:
    """Агент проверки качества результата."""

    MAX_RECOMMENDATIONS = 3

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
            tool_cars_by_id = {car.id: car for car in tool_cars}

            code_reviews, global_issues = self._run_code_checks(
                session=session,
                recommendations=recommendations,
                tool_cars=tool_cars,
            )

            final_reviews = code_reviews
            llm_user_message = ""

            approved_recommendations = [
                recommendation
                for recommendation in recommendations
                if self._is_review_approved(final_reviews, recommendation.car_id)
            ]
            approved_tool_cars = [
                tool_cars_by_id[recommendation.car_id]
                for recommendation in approved_recommendations
                if recommendation.car_id in tool_cars_by_id
            ]

            if approved_recommendations:
                try:
                    llm_result = self._run_llm_checks(
                        session=session,
                        recommendations=approved_recommendations,
                        tool_cars=approved_tool_cars,
                    )
                    final_reviews, llm_issues, llm_user_message = self._merge_llm_result(
                        base_reviews=code_reviews,
                        llm_result=llm_result,
                    )
                    global_issues.extend(llm_issues)
                except ValidationError:
                    final_reviews = code_reviews

            result = self._build_result(
                session=session,
                reviews=final_reviews,
                tool_cars=tool_cars,
                global_issues=global_issues,
                user_message=llm_user_message,
            )

            self.logger.success(
                scenario=scenario_name,
                recommendations_count=len(recommendations),
                approved=result.approved,
                approved_count=len(result.approved_car_ids),
                rejected_count=len(result.rejected_car_ids),
                approved_car_ids=[str(car_id) for car_id in result.approved_car_ids],
                rejected_car_ids=[str(car_id) for car_id in result.rejected_car_ids],
                rejected_issues_by_car=self._rejected_issues_by_car(result.car_reviews),
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
    ) -> tuple[List[CriticCarReview], List[str]]:
        """Проверяет только формальные инварианты, которые не требуют LLM."""

        global_issues: List[str] = []

        if not recommendations:
            global_issues.append("Список рекомендаций пустой.")
            return [], global_issues

        if len(recommendations) > self.MAX_RECOMMENDATIONS:
            global_issues.append(
                f"Агент рекомендовал больше {self.MAX_RECOMMENDATIONS} машин."
            )

        tool_cars_by_id = {car.id: car for car in tool_cars}
        reviews: List[CriticCarReview] = []

        for recommendation in recommendations:
            car_issues: List[str] = []
            car = tool_cars_by_id.get(recommendation.car_id)

            if car is None:
                car_issues.append(
                    f"Машина {recommendation.car_id} не была получена из tools."
                )
            elif session.budget_max is not None and car.price > session.budget_max:
                car_issues.append(
                    f"Машина {car.title()} не попадает в бюджет пользователя."
                )

            reviews.append(
                CriticCarReview(
                    car_id=recommendation.car_id,
                    approved=not car_issues,
                    issues=car_issues,
                )
            )

        return reviews, global_issues

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
            max_output_tokens=1800,
        )

        return CriticResult.model_validate_json(raw_response)

    def _merge_llm_result(
            self,
            base_reviews: List[CriticCarReview],
            llm_result: CriticResult,
    ) -> tuple[List[CriticCarReview], List[str], str]:
        """Объединяет кодовые инварианты и смысловую оценку LLM."""

        llm_reviews_by_id = {
            review.car_id: review
            for review in llm_result.car_reviews
        }
        merged_reviews: List[CriticCarReview] = []

        for base_review in base_reviews:
            if not base_review.approved:
                merged_reviews.append(base_review)
                continue

            llm_review = llm_reviews_by_id.get(base_review.car_id)

            if llm_review is None:
                merged_reviews.append(
                    CriticCarReview(
                        car_id=base_review.car_id,
                        approved=False,
                        issues=["LLM-критик не вернул проверку по этой машине."],
                    )
                )
                continue

            merged_reviews.append(
                CriticCarReview(
                    car_id=base_review.car_id,
                    approved=llm_review.approved,
                    issues=[] if llm_review.approved else list(llm_review.issues),
                )
            )

        return merged_reviews, list(llm_result.issues), llm_result.user_message

    def _build_result(
            self,
            session: UserSession,
            reviews: List[CriticCarReview],
            tool_cars: List[Car],
            global_issues: List[str],
            user_message: str,
    ) -> CriticResult:
        """Нормализует итог проверки и вычисляет агрегированные поля."""

        normalized_reviews = self._normalize_reviews(reviews)
        tool_cars_by_id = {car.id: car for car in tool_cars}

        approved_car_ids = [
            review.car_id
            for review in normalized_reviews
            if review.approved
        ]
        rejected_car_ids = [
            review.car_id
            for review in normalized_reviews
            if not review.approved
        ]
        aggregated_issues = self._collect_result_issues(
            global_issues=global_issues,
            reviews=normalized_reviews,
            tool_cars_by_id=tool_cars_by_id,
        )

        approved = bool(approved_car_ids)

        if len(approved_car_ids) > self.MAX_RECOMMENDATIONS:
            approved = False
            aggregated_issues = self._deduplicate_strings(
                aggregated_issues + [
                    (
                        "После проверки осталось слишком много машин: "
                        f"больше {self.MAX_RECOMMENDATIONS}."
                    )
                ]
            )

        if approved:
            approved_cars = [
                tool_cars_by_id[car_id]
                for car_id in approved_car_ids
                if car_id in tool_cars_by_id
            ]

            if not user_message.strip():
                user_message = self._build_user_message(
                    session=session,
                    tool_cars=approved_cars,
                )
        else:
            user_message = ""

        if not approved and not aggregated_issues:
            aggregated_issues = ["Все рекомендации были отклонены CriticAgent."]

        return CriticResult(
            approved=approved,
            car_reviews=normalized_reviews,
            approved_car_ids=approved_car_ids,
            rejected_car_ids=rejected_car_ids,
            issues=aggregated_issues,
            user_message=user_message,
        )

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
            budget_part = (
                f" Все варианты укладываются в бюджет до {session.budget_max} долларов."
            )

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

    @staticmethod
    def _is_review_approved(reviews: List[CriticCarReview], car_id: UUID) -> bool:
        """Проверяет, была ли машина одобрена на текущем шаге."""

        for review in reviews:
            if review.car_id == car_id:
                return review.approved

        return False

    @staticmethod
    def _normalize_reviews(reviews: List[CriticCarReview]) -> List[CriticCarReview]:
        """Удаляет дубли и приводит reviews к устойчивому виду."""

        normalized_reviews: List[CriticCarReview] = []
        seen_car_ids = set()

        for review in reviews:
            if review.car_id in seen_car_ids:
                continue

            normalized_reviews.append(
                CriticCarReview(
                    car_id=review.car_id,
                    approved=review.approved,
                    issues=[] if review.approved else list(review.issues),
                )
            )
            seen_car_ids.add(review.car_id)

        return normalized_reviews

    def _collect_result_issues(
            self,
            global_issues: List[str],
            reviews: List[CriticCarReview],
            tool_cars_by_id: Dict[UUID, Car],
    ) -> List[str]:
        """Собирает агрегированные проблемы для retry и логов."""

        issues: List[str] = list(global_issues)

        for review in reviews:
            if review.approved:
                continue

            if not review.issues:
                issues.append(
                    self._build_generic_rejection_issue(
                        review=review,
                        tool_cars_by_id=tool_cars_by_id,
                    )
                )
                continue

            for issue in review.issues:
                issues.append(
                    self._with_car_label(
                        review=review,
                        issue=issue,
                        tool_cars_by_id=tool_cars_by_id,
                    )
                )

        return self._deduplicate_strings(issues)

    @staticmethod
    def _build_generic_rejection_issue(
            review: CriticCarReview,
            tool_cars_by_id: Dict[UUID, Car],
    ) -> str:
        """Строит fallback-проблему, если для отклоненной машины не пришли issues."""

        car = tool_cars_by_id.get(review.car_id)
        car_label = car.title() if car is not None else str(review.car_id)
        return f"{car_label}: рекомендация отклонена CriticAgent."

    @staticmethod
    def _with_car_label(
            review: CriticCarReview,
            issue: str,
            tool_cars_by_id: Dict[UUID, Car],
    ) -> str:
        """Добавляет идентификатор машины к проблеме, если его нет в тексте."""

        car = tool_cars_by_id.get(review.car_id)
        car_label = car.title() if car is not None else str(review.car_id)

        if car_label in issue or str(review.car_id) in issue:
            return issue

        return f"{car_label}: {issue}"

    @staticmethod
    def _deduplicate_strings(values: List[str]) -> List[str]:
        """Удаляет дубли строк, сохраняя порядок."""

        result: List[str] = []
        seen = set()

        for value in values:
            if value in seen:
                continue

            result.append(value)
            seen.add(value)

        return result

    @staticmethod
    def _rejected_issues_by_car(reviews: List[CriticCarReview]) -> Dict[str, List[str]]:
        """Готовит компактную структуру логов только по отклоненным машинам."""

        result: Dict[str, List[str]] = {}

        for review in reviews:
            if review.approved:
                continue

            result[str(review.car_id)] = list(review.issues)

        return result
