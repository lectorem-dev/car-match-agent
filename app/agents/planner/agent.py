import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.agents.planner.schemas import PlannedCar, PlannerResult
from app.catalog.car_catalog import CarCatalog
from app.domain.car import Car
from app.domain.user_session import UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class PlannerAgent:
    """Агент подбора автомобилей."""

    MAX_RECOMMENDATIONS = 3
    MAX_CANDIDATES_FOR_LLM = 6

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
        planner_candidates: Optional[List[Car]] = None

        try:
            candidates = self._find_candidates(session=session)
            planner_candidates = candidates[: self.MAX_CANDIDATES_FOR_LLM]

            self.logger.start(
                scenario=scenario_name,
                budget_max=session.budget_max,
                purpose=session.purpose,
                candidates_count=len(planner_candidates),
            )

            if not planner_candidates:
                raise ValueError("Не найдено машин-кандидатов для подбора.")

            try:
                result = self._request_plan(
                    session=session,
                    candidates=planner_candidates,
                    critic_issues=critic_issues or [],
                )
            except (ValidationError, ValueError) as error:
                if not self._is_retryable_planner_error(error):
                    raise

                result = self._request_plan(
                    session=session,
                    candidates=planner_candidates,
                    critic_issues=critic_issues or [],
                    retry_instruction=(
                        "Return only short valid JSON with 1-3 recommendations. "
                        "Do not generate user_message. "
                        "Each reason must be one short phrase."
                    ),
                    max_output_tokens=500,
                )

            result = self._normalize_result(
                result=result,
                candidates=planner_candidates,
            )

            if not result.recommendations or self._should_use_fallback_result(
                session=session,
                result=result,
                candidates=planner_candidates,
            ):
                result = self._build_fallback_result(
                    session=session,
                    candidates=planner_candidates,
                )

            self.logger.success(
                scenario=scenario_name,
                recommendations_count=len(result.recommendations),
                car_ids=[str(item.car_id) for item in result.recommendations],
            )
            return result
        except (ValidationError, ValueError) as error:
            if planner_candidates and self._is_retryable_planner_error(error):
                result = self._build_fallback_result(
                    session=session,
                    candidates=planner_candidates,
                )
                self.logger.success(
                    scenario=scenario_name,
                    recommendations_count=len(result.recommendations),
                    car_ids=[str(item.car_id) for item in result.recommendations],
                )
                return result
            raise
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                has_user_session=session is not None,
                candidates_count=len(planner_candidates) if planner_candidates is not None else None,
                none_object=detect_none_object_name(
                    error,
                    user_session=session,
                    candidate_cars=planner_candidates,
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

        return self._rank_candidates(
            session=session,
            candidates=candidates,
        )

    def _request_plan(
            self,
            session: UserSession,
            candidates: List[Car],
            critic_issues: List[str],
            retry_instruction: Optional[str] = None,
            max_output_tokens: int = 700,
    ) -> PlannerResult:
        """Запрашивает у LLM короткий план рекомендаций."""

        user_prompt = self._build_user_prompt(
            session=session,
            candidates=candidates,
            critic_issues=critic_issues,
            retry_instruction=retry_instruction,
        )

        raw_response = self.llm_client.generate(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=PlannerResult.model_json_schema(),
            response_schema_name="planner_result",
            max_output_tokens=max_output_tokens,
        )

        return PlannerResult.model_validate_json(raw_response)

    def _normalize_result(
            self,
            result: PlannerResult,
            candidates: List[Car],
    ) -> PlannerResult:
        """Оставляет только валидные рекомендации из списка кандидатов."""

        candidates_by_id = {car.id: car for car in candidates}
        normalized_recommendations: List[PlannedCar] = []
        seen_car_ids = set()

        for recommendation in result.recommendations:
            if recommendation.car_id not in candidates_by_id:
                continue

            if recommendation.car_id in seen_car_ids:
                continue

            normalized_recommendations.append(
                PlannedCar(
                    car_id=recommendation.car_id,
                    reason=self._shorten_reason(recommendation.reason),
                    risk_note=None,
                )
            )
            seen_car_ids.add(recommendation.car_id)

            if len(normalized_recommendations) >= self.MAX_RECOMMENDATIONS:
                break

        return PlannerResult(
            recommendations=normalized_recommendations,
            user_message="",
        )

    def _build_fallback_result(
            self,
            session: UserSession,
            candidates: List[Car],
    ) -> PlannerResult:
        """Строит стабильный fallback, если Planner не вернул валидный JSON."""

        fallback_recommendations = [
            PlannedCar(
                car_id=car.id,
                reason=self._build_short_reason(session=session, car=car),
                risk_note=None,
            )
            for car in candidates[: self.MAX_RECOMMENDATIONS]
        ]

        return PlannerResult(
            recommendations=fallback_recommendations,
            user_message="",
        )

    def _rank_candidates(
            self,
            session: UserSession,
            candidates: List[Car],
    ) -> List[Car]:
        """Сортирует кандидатов по простым локальным сигналам без внешних знаний."""

        return sorted(
            candidates,
            key=lambda car: self._candidate_score(session=session, car=car),
            reverse=True,
        )

    @staticmethod
    def _is_retryable_planner_error(error: Exception) -> bool:
        """Определяет, можно ли перейти к повторной попытке или fallback."""

        if isinstance(error, ValidationError):
            return True

        return str(error) == "Модель вернула пустой ответ"

    def _build_user_prompt(
            self,
            session: UserSession,
            candidates: List[Car],
            critic_issues: List[str],
            retry_instruction: Optional[str] = None,
    ) -> str:
        """Собирает вход для LLM."""

        payload = {
            "user_session": self._session_to_dict(session),
            "candidate_cars": [self._car_to_dict(car) for car in candidates],
            "critic_issues": critic_issues,
            "output_rule": "Choose 1-3 cars only from candidate_cars. Return valid JSON only.",
            "currency_rule": "All prices and budgets are in USD.",
        }

        if retry_instruction is not None:
            payload["retry_instruction"] = retry_instruction

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

    def _build_short_reason(self, session: UserSession, car: Car) -> str:
        """Строит короткую машинную причину выбора без внешних знаний."""

        reason_parts: List[str] = []
        context_text = self._build_context_text(session)

        if session.budget_max is not None and car.price <= session.budget_max:
            reason_parts.append("fits budget")

        if self._contains_any(context_text, ["automatic", "автомат"]):
            if car.transmission.value in {"automatic", "cvt", "robot"}:
                reason_parts.append(car.transmission.value)

        if self._contains_any(context_text, ["awd", "full drive", "полный привод"]) and car.drive_type.value == "awd":
            reason_parts.append("awd")

        if self._contains_any(context_text, ["electric", "электро"]) and car.fuel_type.value == "electric":
            reason_parts.append("electric")

        if self._contains_any(context_text, ["hybrid", "гибрид"]) and car.fuel_type.value == "hybrid":
            reason_parts.append("hybrid")

        if session.preferred_body_types and car.body_type.value in session.preferred_body_types:
            reason_parts.append(car.body_type.value)

        if not reason_parts:
            reason_parts.append("best available match")

        return self._shorten_reason(" + ".join(reason_parts))

    @staticmethod
    def _normalize_terms(values: List[str]) -> List[str]:
        """Нормализует текстовые требования для коротких локальных причин."""

        return [value.strip().lower() for value in values if value.strip()]

    @staticmethod
    def _shorten_reason(reason: str) -> str:
        """Ограничивает длину reason одной короткой фразой."""

        compact_reason = " ".join(reason.strip().split())

        if len(compact_reason) <= 80:
            return compact_reason

        return compact_reason[:77].rstrip() + "..."

    def _candidate_score(self, session: UserSession, car: Car) -> int:
        """Оценивает, насколько машина подходит текущей сессии."""

        score = 0
        normalized_body_types = self._normalize_terms(session.preferred_body_types)
        normalized_brands = self._normalize_terms(session.preferred_brands)
        context_text = self._build_context_text(session)

        requires_automatic = self._contains_any(context_text, ["automatic", "автомат"])
        requires_awd = self._contains_any(context_text, ["awd", "full drive", "полный привод"])
        needs_electric = self._contains_any(context_text, ["electric", "электро", "электромоб"])
        prefers_hybrid = self._contains_any(context_text, ["hybrid", "гибрид"])
        needs_family_car = self._contains_any(context_text, ["family", "сем", "дет"])
        needs_city_car = self._contains_any(context_text, ["city", "город", "park", "парков"])
        needs_winter_or_rough_roads = self._contains_any(
            context_text,
            ["winter", "зим", "rough road", "плох", "dacha", "дач", "за город", "загород"],
        )
        needs_business_or_highway = self._contains_any(
            context_text,
            ["business", "делов", "represent", "представ", "highway", "трасс"],
        )
        needs_large_trunk = self._contains_any(
            context_text,
            ["багаж", "trunk", "luggage", "вмест", "practical", "практич"],
        )
        prefers_comfort = self._contains_any(context_text, ["comfort", "комфорт"])
        prefers_prestige = self._contains_any(context_text, ["prestige", "престиж"])
        prefers_long_distance_efficiency = self._contains_any(
            context_text,
            ["economical", "эконом", "long distance", "дальн", "high mileage", "больших пробег"],
        )

        if session.budget_max is not None and car.price <= session.budget_max:
            score += 3

        if car.brand.lower() in normalized_brands:
            score += 3

        if car.body_type.value in normalized_body_types:
            score += 3

        if requires_automatic and car.transmission.value == "automatic":
            score += 4
        elif requires_automatic and car.transmission.value in {"cvt", "robot"}:
            score += 3
        elif requires_automatic and car.transmission.value == "manual":
            score -= 4

        if requires_awd and car.drive_type.value == "awd":
            score += 8
        elif requires_awd:
            score -= 6

        if needs_electric and car.fuel_type.value == "electric":
            score += 10
        elif needs_electric:
            score -= 8

        if prefers_hybrid and car.fuel_type.value == "hybrid":
            score += 6

        if needs_family_car or session.family_size is not None and session.family_size >= 4:
            if car.body_type.value == "minivan":
                score += 6
            elif car.body_type.value == "suv":
                score += 4
            elif car.body_type.value in {"wagon", "liftback"}:
                score += 3

        if prefers_comfort:
            if car.body_type.value in {"sedan", "suv", "minivan", "wagon", "liftback"}:
                score += 2
            if car.year >= 2019:
                score += 1

        if needs_city_car:
            if car.body_type.value in {"hatchback", "sedan", "liftback"}:
                score += 2

        if needs_winter_or_rough_roads:
            if car.drive_type.value == "awd":
                score += 5
            if car.body_type.value == "suv":
                score += 4
            if car.body_type.value == "pickup":
                score += 3

        if needs_large_trunk:
            if car.body_type.value == "minivan":
                score += 6
            elif car.body_type.value in {"wagon", "liftback"}:
                score += 5
            elif car.body_type.value == "suv":
                score += 2

        if needs_business_or_highway:
            if car.body_type.value in {"sedan", "wagon", "liftback"}:
                score += 5
            elif car.body_type.value == "suv":
                score += 1

        if prefers_prestige and car.brand.lower() in {"mercedes-benz", "bmw", "audi", "lexus"}:
            score += 5

        if prefers_long_distance_efficiency:
            if car.fuel_type.value == "diesel":
                score += 5
            elif car.fuel_type.value == "hybrid":
                score += 3
            if car.body_type.value in {"sedan", "wagon", "liftback"}:
                score += 1

        if "beginner" == (session.experience_level or "").lower() and car.transmission.value in {"automatic", "cvt"}:
            score += 1

        if car.year >= 2020:
            score += 2
        elif car.year >= 2018:
            score += 1

        if car.mileage_km > 120000:
            score -= 2
        elif car.mileage_km > 80000:
            score -= 1

        return score

    def _should_use_fallback_result(
            self,
            session: UserSession,
            result: PlannerResult,
            candidates: List[Car],
    ) -> bool:
        """Возвращает fallback, если LLM выбрал заметно худший набор машин."""

        if not result.recommendations:
            return True

        candidates_by_id = {car.id: car for car in candidates}
        selected_cars = [
            candidates_by_id[item.car_id]
            for item in result.recommendations
            if item.car_id in candidates_by_id
        ]

        if not selected_cars:
            return True

        comparison_count = min(len(selected_cars), self.MAX_RECOMMENDATIONS)
        top_cars = candidates[:comparison_count]

        selected_score = sum(
            self._candidate_score(session=session, car=car)
            for car in selected_cars[:comparison_count]
        )
        top_score = sum(
            self._candidate_score(session=session, car=car)
            for car in top_cars
        )

        return selected_score + 1 < top_score

    @staticmethod
    def _build_context_text(session: UserSession) -> str:
        """Собирает единый текстовый контекст для эвристик."""

        parts = [
            session.purpose or "",
            session.user_notes,
            " ".join(session.must_have),
            " ".join(session.preferred_body_types),
            " ".join(session.preferred_brands),
        ]
        return " ".join(parts).lower()

    @staticmethod
    def _contains_any(text: str, patterns: List[str]) -> bool:
        """Проверяет, содержит ли текст хотя бы один из паттернов."""

        return any(pattern in text for pattern in patterns)
