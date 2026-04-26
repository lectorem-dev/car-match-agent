from dataclasses import dataclass
from typing import Optional, Protocol

from app.domain.user_session import UserSession
from app.evals.judge import ScenarioJudge
from app.evals.schemas import ScenarioStepType, TestScenario
from app.orchestrator.schemas import PipelineResponse
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class AgentLike(Protocol):
    """Минимальный интерфейс агента для eval-runner."""

    def handle_message(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool,
            scenario_name: Optional[str] = None,
    ) -> PipelineResponse:
        """Обрабатывает сообщение пользователя и возвращает структурированный ответ."""
        ...


@dataclass
class ScenarioRunResult:
    """Результат прогона одного сценария."""

    scenario_id: str
    scenario_name: str
    passed: bool
    error: Optional[str] = None


class ScenarioRunner:
    """Прогоняет агента по тестовому сценарию."""

    def __init__(
            self,
            agent: AgentLike,
            judge: Optional[ScenarioJudge] = None,
            enable_logs: bool = True,
    ) -> None:
        self.agent = agent
        self.judge = judge or ScenarioJudge()
        self.logger = AgentLogger(
            "Pipeline",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_WHITE,
        )

    def run(self, scenario: TestScenario) -> ScenarioRunResult:
        """Запускает один сценарий целиком."""
        session = scenario.initial_session or UserSession()
        last_response: Optional[PipelineResponse] = None
        current_step_index: Optional[int] = None
        current_step_type: Optional[str] = None
        current_step_user_message: Optional[str] = None
        current_step_selected_car_id: Optional[str] = None

        self.logger.event(
            "scenario_start",
            scenario=scenario.id,
            title=scenario.name,
        )

        try:
            for step_index, step in enumerate(scenario.steps, start=1):
                current_step_index = step_index
                current_step_type = step.step_type.value
                current_step_user_message = step.user_message
                current_step_selected_car_id = str(step.selected_car_id) if step.selected_car_id else None

                if step.step_type == ScenarioStepType.USER_MESSAGE:
                    if step.user_message is None:
                        raise AssertionError("Для шага USER_MESSAGE нужен user_message.")

                    last_response = self.agent.handle_message(
                        user_message=step.user_message,
                        session=session,
                        allow_clarifying_question=scenario.allow_clarifying_question,
                        scenario_name=scenario.id,
                    )

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                        session=session,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_CLARIFYING_QUESTION:
                    if last_response is None:
                        raise AssertionError("Нет ответа агента для проверки уточняющего вопроса.")

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                        session=session,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_RECOMMENDATION:
                    if last_response is None:
                        raise AssertionError("Нет ответа агента для проверки рекомендации.")

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                        session=session,
                    )

                elif step.step_type == ScenarioStepType.USER_SELECT_CAR:
                    if step.selected_car_id is None:
                        raise AssertionError("Для шага USER_SELECT_CAR нужен selected_car_id.")

                    session.select_car(step.selected_car_id)

                    self.judge.check_selected_car(
                        session=session,
                        expected_car_id=step.selected_car_id,
                    )

                elif step.step_type == ScenarioStepType.USER_REQUEST_RESERVATION:
                    if step.user_message is None:
                        raise AssertionError("Для шага USER_REQUEST_RESERVATION нужен user_message.")

                    last_response = self.agent.handle_message(
                        user_message=step.user_message,
                        session=session,
                        allow_clarifying_question=scenario.allow_clarifying_question,
                        scenario_name=scenario.id,
                    )

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                        session=session,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_RESERVATION_CREATED:
                    self.judge.check_reservation_created(session=session)

                else:
                    raise AssertionError(f"Неизвестный тип шага: {step.step_type}")

        except Exception as error:
            self.logger.event(
                "scenario_fail",
                scenario=scenario.id,
                step_index=current_step_index,
                step_type=current_step_type,
                user_message=current_step_user_message,
                selected_car_id=current_step_selected_car_id,
                error_type=error.__class__.__name__,
                error=str(error),
                none_object=detect_none_object_name(
                    error,
                    session=session,
                    last_response=last_response,
                ),
            )
            return ScenarioRunResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                passed=False,
                error=str(error),
            )

        self.logger.event(
            "scenario_success",
            scenario=scenario.id,
        )

        return ScenarioRunResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            passed=True,
        )
