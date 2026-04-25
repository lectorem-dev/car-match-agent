import logging
from typing import List, Optional

from app.agents.critic.agent import CriticAgent
from app.agents.domain_guard.agent import DomainGuardAgent
from app.agents.domain_guard.schemas import DomainStatus
from app.agents.extractor.agent import Extractor
from app.agents.planner.agent import PlannerAgent
from app.agents.planner.schemas import PlannerResult
from app.agents.reservation.agent import ReservationAgent
from app.agents.reservation.schemas import ReservationStatus
from app.catalog.car_catalog import CarCatalog
from app.domain.agent_response import AgentResponse, CarRecommendation, SessionUpdate
from app.domain.car import Car
from app.domain.user_session import DialogStatus, UserSession
from app.services.session_update_service import SessionUpdateService


# Цвет логов оркестратора.
PIPELINE_LOG_COLOR = "\033[95m"
PIPELINE_LOG_RESET = "\033[0m"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("car_match_agent.pipeline")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[pipeline] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


LOGGER = _build_logger()


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

    def handle_message(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool = True,
    ) -> AgentResponse:
        """Обрабатывает сообщение пользователя через весь пайплайн."""

        self._log("Новое сообщение пользователя")
        self._log("message=%s", user_message)

        domain_result = self.domain_guard.check(
            user_message=user_message,
            session=session,
        )

        self._log("domain_status=%s", domain_result.domain_status.value)

        if domain_result.domain_status == DomainStatus.OUT_OF_DOMAIN:
            self._log("Запрос вне домена, пайплайн остановлен")

            return AgentResponse(
                user_message=domain_result.user_message,
                should_ask_clarifying_question=False,
                recommended_cars=[],
            )

        reservation_result = self.reservation.handle(
            user_message=user_message,
            session=session,
        )

        self._log("reservation_status=%s", reservation_result.status.value)

        if reservation_result.status != ReservationStatus.NOT_RESERVATION:
            self._log("Сообщение обработано ReservationAgent")

            return self._reservation_result_to_agent_response(
                reservation_result=reservation_result,
            )

        extraction = self.extractor.extract(
            user_message=user_message,
            session=session,
            allow_clarifying_question=allow_clarifying_question,
        )

        self._log("requirements_extracted=true")

        session = self.session_update_service.apply_update(
            session=session,
            update=extraction.session_update,
        )

        self._try_select_car_by_title(
            session=session,
            selected_car_title=extraction.selected_car_title,
        )

        self._log("session_budget_max=%s", session.budget_max)
        self._log("session_purpose=%s", session.purpose)

        if not session.has_required_data_for_recommendation():
            return self._handle_missing_required_data(
                session=session,
                extraction=extraction,
                allow_clarifying_question=allow_clarifying_question,
            )

        return self._handle_recommendation(
            session=session,
            session_update=extraction.session_update,
        )

    def _handle_missing_required_data(
            self,
            session: UserSession,
            extraction,
            allow_clarifying_question: bool,
    ) -> AgentResponse:
        """Обрабатывает ситуацию, когда данных для подбора мало."""

        self._log("required_data_ready=false")

        if allow_clarifying_question:
            question = extraction.clarifying_question or self._build_missing_data_question(session=session)
            session.dialog_status = DialogStatus.CLARIFYING_QUESTION

            self._log("clarifying_question=true")

            return AgentResponse(
                user_message=question,
                session_update=extraction.session_update,
                should_ask_clarifying_question=True,
                clarifying_question=question,
            )

        if session.purpose is None:
            session.purpose = "buy suitable car"
            self._log("default_purpose_applied=%s", session.purpose)

        if session.has_required_data_for_recommendation():
            return self._handle_recommendation(
                session=session,
                session_update=extraction.session_update,
            )

        self._log("Пайплайн остановлен: недостаточно данных")

        return AgentResponse(
            user_message=(
                "Для подбора машины нужен минимум: бюджет и цель покупки. "
                "В этом сценарии уточняющий вопрос отключен, поэтому рекомендация не выполняется."
            ),
            session_update=extraction.session_update,
            should_ask_clarifying_question=False,
            recommended_cars=[],
        )

    def _handle_recommendation(
            self,
            session: UserSession,
            session_update: SessionUpdate,
    ) -> AgentResponse:
        """Запускает Planner и Critic."""

        self._log("recommendation_started=true")

        critic_issues: List[str] = []

        for attempt in range(self.max_planner_retries + 1):
            self._log("planner_attempt=%s", attempt + 1)

            planner_result = self.planner.plan(
                session=session,
                critic_issues=critic_issues,
            )

            tool_cars = self._load_recommended_tool_cars(
                planner_result=planner_result,
            )

            recommendations = [
                CarRecommendation(
                    car_id=item.car_id,
                    reason=item.reason,
                    risk_note=item.risk_note,
                )
                for item in planner_result.recommendations
            ]

            critic_result = self.critic.review(
                session=session,
                recommendations=recommendations,
                tool_cars=tool_cars,
                user_message=planner_result.user_message,
            )

            self._log("critic_approved=%s", critic_result.approved)

            if critic_result.approved:
                session.dialog_status = DialogStatus.READY_TO_RECOMMEND

                self._log("recommendation_finished=true")
                self._log("recommended_cars_count=%s", len(recommendations))

                return AgentResponse(
                    user_message=planner_result.user_message,
                    session_update=session_update,
                    should_ask_clarifying_question=False,
                    recommended_cars=recommendations,
                )

            critic_issues = critic_result.issues
            self._log("critic_issues=%s", "; ".join(critic_issues))

        self._log("recommendation_failed=true")

        return AgentResponse(
            user_message=(
                    "Не удалось подготовить корректную рекомендацию. "
                    "Проблемы проверки: " + "; ".join(critic_issues)
            ),
            session_update=session_update,
            should_ask_clarifying_question=False,
            recommended_cars=[],
        )

    def _reservation_result_to_agent_response(
            self,
            reservation_result,
    ) -> AgentResponse:
        """Преобразует результат бронирования в AgentResponse."""

        return AgentResponse(
            user_message=reservation_result.user_message,
            should_ask_clarifying_question=False,
            recommended_cars=[],
            selected_car_id=reservation_result.selected_car_id,
            ready_for_reservation=reservation_result.status == ReservationStatus.CREATED,
        )

    def _build_missing_data_question(self, session: UserSession) -> str:
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
    ) -> None:
        """Пытается выбрать машину по названию из сообщения пользователя."""

        if not selected_car_title:
            return

        selected_car_title_lower = selected_car_title.lower()

        for car in self.catalog.find_all():
            if car.title().lower() == selected_car_title_lower:
                session.select_car(car.id)
                self._log("selected_car_id=%s", car.id)
                return

            short_title = f"{car.brand} {car.model}".lower()

            if short_title == selected_car_title_lower:
                session.select_car(car.id)
                self._log("selected_car_id=%s", car.id)
                return

    def _load_recommended_tool_cars(self, planner_result: PlannerResult) -> List[Car]:
        """Загружает машины из каталога по id рекомендаций."""

        result: List[Car] = []

        for recommendation in planner_result.recommendations:
            car = self.catalog.find_by_id(recommendation.car_id)

            if car is not None:
                result.append(car)

        return result

    def _log(self, message: str, *args) -> None:
        """Пишет лог пайплайна, если логирование включено."""

        if not self.enable_logging:
            return

        LOGGER.info(f"{PIPELINE_LOG_COLOR}{message}{PIPELINE_LOG_RESET}", *args)