from dataclasses import dataclass
from typing import Optional, Protocol

from app.domain.agent_response import AgentResponse
from app.domain.user_session import UserSession
from app.evals.judge import ScenarioJudge
from app.evals.schemas import ScenarioStepType, TestScenario


class AgentLike(Protocol):
    """Минимальный интерфейс агента для eval-runner."""

    def handle_message(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool,
    ) -> AgentResponse:
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
    ) -> None:
        self.agent = agent
        self.judge = judge or ScenarioJudge()

    def run(self, scenario: TestScenario) -> ScenarioRunResult:
        """Запускает один сценарий целиком."""
        session = scenario.initial_session or UserSession()
        last_response: Optional[AgentResponse] = None

        try:
            for step in scenario.steps:
                if step.step_type == ScenarioStepType.USER_MESSAGE:
                    if step.user_message is None:
                        raise AssertionError("Для шага USER_MESSAGE нужен user_message.")

                    last_response = self.agent.handle_message(
                        user_message=step.user_message,
                        session=session,
                        allow_clarifying_question=scenario.allow_clarifying_question,
                    )

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_CLARIFYING_QUESTION:
                    if last_response is None:
                        raise AssertionError("Нет ответа агента для проверки уточняющего вопроса.")

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_RECOMMENDATION:
                    if last_response is None:
                        raise AssertionError("Нет ответа агента для проверки рекомендации.")

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
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
                    )

                    self.judge.check_step_response(
                        scenario=scenario,
                        step=step,
                        response=last_response,
                    )

                elif step.step_type == ScenarioStepType.EXPECT_RESERVATION_CREATED:
                    self.judge.check_reservation_created(session=session)

                else:
                    raise AssertionError(f"Неизвестный тип шага: {step.step_type}")

        except Exception as error:
            return ScenarioRunResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                passed=False,
                error=str(error),
            )

        return ScenarioRunResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            passed=True,
        )
