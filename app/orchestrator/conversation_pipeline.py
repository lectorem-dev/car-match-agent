from typing import Dict, List, Optional
from uuid import UUID

from app.agents.critic.critic_agent import CriticAgent
from app.agents.critic.critic_schemas import CriticCarReview
from app.agents.guard.guard_agent import DomainGuardAgent
from app.agents.guard.guard_schemas import DomainStatus
from app.agents.extractor.extractor_agent import Extractor
from app.agents.extractor.extractor_schemas import ExtractorResult
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.planner.planner_schemas import PlannerResult
from app.agents.reservation.reservation_agent import ReservationAgent
from app.agents.reservation.reservation_schemas import (
    ReservationCreatedResult,
    ReservationIntent,
)
from app.agent_tools.car_catalog import CarCatalog
from app.domain.car import Car
from app.domain.user_session import DialogStatus, UserSession
from app.orchestrator.orchestrator_schemas import PipelineResponse, RecommendedCar
from app.session.session_update_service import SessionUpdateService
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class Pipeline:
    """Кодовый оркестратор агентного пайплайна."""

    def __init__(
            self,
            domain_guard: DomainGuardAgent,
            reservation: ReservationAgent,
            extractor: Extractor,
            planner: PlannerAgent,
            critic: CriticAgent,
            catalog: CarCatalog,
            session_update_service: SessionUpdateService,
            max_planner_retries: int = 1,
            enable_logging: bool = True,
    ) -> None:
        self.domain_guard = domain_guard
        self.reservation = reservation
        self.extractor = extractor
        self.planner = planner
        self.critic = critic
        self.catalog = catalog
        self.session_update_service = session_update_service
        self.max_planner_retries = max_planner_retries
        self.enable_logging = enable_logging
        self.logger = AgentLogger(
            "Pipeline",
            enabled=enable_logging,
            color=AgentLogColor.BRIGHT_WHITE,
        )

    def handle_message(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool = True,
            scenario_name: Optional[str] = None,
    ) -> PipelineResponse:
        """Обрабатывает сообщение пользователя через весь пайплайн."""

        domain_result = None
        reservation_result = None
        extraction = None

        try:
            domain_result = self.domain_guard.check(
                user_message=user_message,
                session=session,
                scenario_name=scenario_name,
            )

            self.logger.state(
                scenario=scenario_name,
                step="domain_guard_done",
                domain_status=domain_result.domain_status.value,
            )

            if domain_result.domain_status == DomainStatus.OUT_OF_DOMAIN:
                self.logger.decision(
                    "out_of_domain",
                    scenario=scenario_name,
                    domain_status=domain_result.domain_status.value,
                )

                return PipelineResponse(
                    user_message=domain_result.user_message,
                )

            reservation_result = self.reservation.handle(
                user_message=user_message,
                session=session,
                scenario_name=scenario_name,
            )

            self.logger.state(
                scenario=scenario_name,
                step="reservation_check_done",
                intent=reservation_result.intent.value,
                selected_car_title=reservation_result.selected_car_title,
                current_selected_car_id=self._selected_car_id_value(session),
            )

            if reservation_result.intent == ReservationIntent.RESERVATION_REQUEST:
                return self._handle_reservation_request(
                    session=session,
                    scenario_name=scenario_name,
                )

            extraction = self.extractor.extract(
                user_message=user_message,
                session=session,
                allow_clarifying_question=allow_clarifying_question,
                scenario_name=scenario_name,
            )

            self.logger.state(
                scenario=scenario_name,
                step="extractor_done",
                has_extractor_result=extraction is not None,
                has_session_update=extraction.session_update is not None if extraction else False,
            )

            session = self.session_update_service.apply_update(
                session=session,
                update=extraction.session_update,
            )

            self.logger.state(
                scenario=scenario_name,
                step="session_update_applied",
                has_session=session is not None,
                budget_max=session.budget_max if session else None,
                purpose=session.purpose if session else None,
                family_size=session.family_size if session else None,
                must_have=session.must_have if session else None,
                dialog_status=self._dialog_status_value(session),
                current_selected_car_id=self._selected_car_id_value(session),
            )

            self._try_select_car_by_title(
                session=session,
                selected_car_title=extraction.selected_car_title,
                scenario_name=scenario_name,
            )

            self.logger.state(
                scenario=scenario_name,
                step="required_data_check",
                has_session=session is not None,
                budget_max=session.budget_max if session else None,
                purpose=session.purpose if session else None,
            )

            if session is not None:
                self.logger.decision(
                    "ready_to_recommend"
                    if session.has_required_data_for_recommendation()
                    else "not_ready_to_recommend",
                    scenario=scenario_name,
                    budget_max=session.budget_max,
                    purpose=session.purpose,
                )
            else:
                self.logger.state(
                    scenario=scenario_name,
                    step="required_data_decision_skipped",
                    has_session=False,
                )

            if not session.has_required_data_for_recommendation():
                return self._handle_missing_required_data(
                    session=session,
                    extraction=extraction,
                    allow_clarifying_question=allow_clarifying_question,
                    scenario_name=scenario_name,
                )

            return self._handle_recommendation(
                session=session,
                scenario_name=scenario_name,
            )
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                user_message=user_message,
                has_session=session is not None,
                dialog_status=self._dialog_status_value(session),
                has_domain_result=domain_result is not None,
                has_reservation_result=reservation_result is not None,
                has_extractor_result=extraction is not None,
                none_object=detect_none_object_name(
                    error,
                    session=session,
                    domain_result=domain_result,
                    reservation_result=reservation_result,
                    extractor_result=extraction,
                ),
            )
            raise

    def _handle_missing_required_data(
            self,
            session: UserSession,
            extraction: ExtractorResult,
            allow_clarifying_question: bool,
            scenario_name: Optional[str] = None,
    ) -> PipelineResponse:
        """Обрабатывает ситуацию, когда данных для подбора мало."""

        self.logger.state(
            scenario=scenario_name,
            step="missing_required_data",
            allow_clarifying_question=allow_clarifying_question,
        )

        if allow_clarifying_question:
            question = extraction.clarifying_question or self._build_missing_data_question(session=session)
            session.dialog_status = DialogStatus.CLARIFYING_QUESTION

            self.logger.state(
                scenario=scenario_name,
                step="clarifying_question_prepared",
                dialog_status=session.dialog_status.value,
            )

            return PipelineResponse(
                user_message=question,
                should_ask_clarifying_question=True,
                clarifying_question=question,
            )

        if session.purpose is None:
            session.purpose = "buy suitable car"
            self.logger.state(
                scenario=scenario_name,
                step="default_purpose_applied",
                purpose=session.purpose,
            )

        if session.has_required_data_for_recommendation():
            return self._handle_recommendation(
                session=session,
                scenario_name=scenario_name,
            )

        return PipelineResponse(
            user_message=(
                "Для подбора машины нужен минимум: бюджет и цель покупки. "
                "В этом сценарии уточняющий вопрос отключен, поэтому рекомендация не выполняется."
            ),
        )

    def _handle_recommendation(
            self,
            session: UserSession,
            scenario_name: Optional[str] = None,
    ) -> PipelineResponse:
        """Запускает Planner и Critic."""

        self.logger.state(
            scenario=scenario_name,
            step="recommendation_started",
        )

        critic_issues: List[str] = []

        for attempt in range(self.max_planner_retries + 1):
            self.logger.state(
                scenario=scenario_name,
                step="planner_attempt",
                attempt=attempt + 1,
            )

            planner_result = self.planner.plan(
                session=session,
                critic_issues=critic_issues,
                scenario_name=scenario_name,
            )

            tool_cars = self._load_recommended_tool_cars(
                planner_result=planner_result,
            )

            recommendations = [
                RecommendedCar(
                    car_id=item.car_id,
                    reason=item.reason,
                    risk_note=item.risk_note,
                )
                for item in planner_result.recommendations
            ]

            critic_result = self.critic.review(
                session=session,
                recommendations=planner_result.recommendations,
                tool_cars=tool_cars,
                scenario_name=scenario_name,
            )

            approved_recommendations = self._filter_recommendations_by_ids(
                recommendations=recommendations,
                approved_car_ids=critic_result.approved_car_ids,
            )

            self.logger.state(
                scenario=scenario_name,
                step="critic_review_done",
                recommendations_count=len(recommendations),
                approved_count=len(critic_result.approved_car_ids),
                rejected_count=len(critic_result.rejected_car_ids),
                approved_car_ids=[str(car_id) for car_id in critic_result.approved_car_ids],
                rejected_car_ids=[str(car_id) for car_id in critic_result.rejected_car_ids],
                rejected_issues_by_car=self._rejected_issues_by_car(critic_result.car_reviews),
            )

            if critic_result.approved and approved_recommendations:
                if session.dialog_status not in {
                    DialogStatus.CAR_SELECTED,
                    DialogStatus.READY_FOR_RESERVATION,
                    DialogStatus.RESERVATION_CREATED,
                }:
                    session.selected_car_id = approved_recommendations[0].car_id

                session.dialog_status = DialogStatus.READY_TO_RECOMMEND

                self.logger.state(
                    scenario=scenario_name,
                    step="recommendation_ready",
                    recommended_cars_count=len(approved_recommendations),
                    approved_count=len(critic_result.approved_car_ids),
                    rejected_count=len(critic_result.rejected_car_ids),
                    approved_car_ids=[str(car_id) for car_id in critic_result.approved_car_ids],
                    rejected_car_ids=[str(car_id) for car_id in critic_result.rejected_car_ids],
                    rejected_issues_by_car=self._rejected_issues_by_car(critic_result.car_reviews),
                    current_selected_car_id=self._selected_car_id_value(session),
                    dialog_status=session.dialog_status.value,
                )

                return PipelineResponse(
                    user_message=critic_result.user_message,
                    recommended_cars=approved_recommendations,
                )

            critic_issues = critic_result.issues
            self.logger.state(
                scenario=scenario_name,
                step="critic_rejected",
                recommendations_count=len(recommendations),
                approved_count=len(critic_result.approved_car_ids),
                rejected_count=len(critic_result.rejected_car_ids),
                approved_car_ids=[str(car_id) for car_id in critic_result.approved_car_ids],
                rejected_car_ids=[str(car_id) for car_id in critic_result.rejected_car_ids],
                rejected_issues_by_car=self._rejected_issues_by_car(critic_result.car_reviews),
                issues=critic_issues,
            )

        return PipelineResponse(
            user_message=(
                    "Не удалось подготовить корректную рекомендацию. "
                    "Проблемы проверки: " + "; ".join(critic_issues)
            ),
        )

    def _handle_reservation_request(
            self,
            session: UserSession,
            scenario_name: Optional[str] = None,
    ) -> PipelineResponse:
        """Обрабатывает запрос на бронирование после определения intent."""

        if session.selected_car_id is None:
            self.logger.state(
                scenario=scenario_name,
                step="reservation_request_processed",
                reservation_status="need_car_selection",
                current_selected_car_id=self._selected_car_id_value(session),
                dialog_status=self._dialog_status_value(session),
            )

            return PipelineResponse(
                user_message="Сначала нужно выбрать конкретную машину, затем я создам заявку на бронь.",
            )

        selected_car = self.catalog.find_by_id(session.selected_car_id)

        if selected_car is None:
            self.logger.state(
                scenario=scenario_name,
                step="reservation_request_processed",
                reservation_status="car_not_found",
                current_selected_car_id=self._selected_car_id_value(session),
                dialog_status=self._dialog_status_value(session),
            )

            return PipelineResponse(
                user_message="Выбранная машина не найдена в каталоге. Нужно выбрать другой автомобиль.",
            )

        session.mark_reservation_created()
        created_result = ReservationCreatedResult(
            user_message=f"Mock-заявка на бронь создана: {selected_car.title()}.",
        )

        self.logger.state(
            scenario=scenario_name,
            step="reservation_request_processed",
            reservation_status="created",
            current_selected_car_id=str(selected_car.id),
            dialog_status=self._dialog_status_value(session),
        )

        return PipelineResponse(
            user_message=created_result.user_message,
            selected_car_id=selected_car.id,
            ready_for_reservation=created_result.ready_for_reservation,
        )

    @staticmethod
    def _build_missing_data_question(session: UserSession) -> str:
        """Собирает уточняющий вопрос по недостающим данным."""

        if session.budget_max is None and session.purpose is None:
            return "Уточните бюджет и цель покупки: для города, семьи, работы или поездок?"

        if session.budget_max is None:
            return "Уточните бюджет: до какой суммы в долларах подбирать машину?"

        return "Уточните цель покупки: для города, семьи, работы или поездок?"

    def _try_select_car_by_title(
            self,
            session: UserSession,
            selected_car_title: Optional[str],
            scenario_name: Optional[str] = None,
    ) -> None:
        """Пытается выбрать машину по названию из сообщения пользователя."""

        if not selected_car_title:
            return

        selected_car_title_lower = selected_car_title.lower()

        for car in self.catalog.find_all():
            if car.title().lower() == selected_car_title_lower:
                session.select_car(car.id)
                self.logger.state(
                    scenario=scenario_name,
                    step="selected_car_title_resolved",
                    selected_car_title=selected_car_title,
                    current_selected_car_id=str(car.id),
                    dialog_status=self._dialog_status_value(session),
                )
                return

            short_title = f"{car.brand} {car.model}".lower()

            if short_title == selected_car_title_lower:
                session.select_car(car.id)
                self.logger.state(
                    scenario=scenario_name,
                    step="selected_car_title_resolved",
                    selected_car_title=selected_car_title,
                    current_selected_car_id=str(car.id),
                    dialog_status=self._dialog_status_value(session),
                )
                return

    def _load_recommended_tool_cars(self, planner_result: PlannerResult) -> List[Car]:
        """Загружает машины из каталога по id рекомендаций."""

        result: List[Car] = []

        for recommendation in planner_result.recommendations:
            car = self.catalog.find_by_id(recommendation.car_id)

            if car is not None:
                result.append(car)

        return result

    @staticmethod
    def _filter_recommendations_by_ids(
            recommendations: List[RecommendedCar],
            approved_car_ids: List[UUID],
    ) -> List[RecommendedCar]:
        """Оставляет только те рекомендации, которые одобрил CriticAgent."""

        approved_car_ids_set = set(approved_car_ids)

        return [
            recommendation
            for recommendation in recommendations
            if recommendation.car_id in approved_car_ids_set
        ]

    @staticmethod
    def _rejected_issues_by_car(car_reviews: List[CriticCarReview]) -> Dict[str, List[str]]:
        """Собирает issues только по отклоненным машинам для логов."""

        result: Dict[str, List[str]] = {}

        for review in car_reviews:
            if review.approved:
                continue

            result[str(review.car_id)] = list(review.issues)

        return result

    @staticmethod
    def _dialog_status_value(session: Optional[UserSession]) -> Optional[str]:
        """Возвращает строковое значение dialog_status, если сессия доступна."""

        if session is None:
            return None

        return session.dialog_status.value

    @staticmethod
    def _selected_car_id_value(session: Optional[UserSession]) -> Optional[str]:
        """Возвращает selected_car_id как строку, если он есть."""

        if session is None or session.selected_car_id is None:
            return None

        return str(session.selected_car_id)
