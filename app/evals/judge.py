from typing import List
from uuid import UUID

from app.domain.agent_response import AgentResponse
from app.domain.user_session import DialogStatus, UserSession
from app.evals.schemas import ScenarioStep, ScenarioStepType, TestScenario


class ScenarioJudge:
    """Проверяет ответы агента и состояние сессии по шагам сценария."""

    def check_no_forbidden_clarification(
            self,
            scenario: TestScenario,
            response: AgentResponse,
    ) -> None:
        """Проверяет, что агент не задал уточняющий вопрос, если это запрещено."""
        if not scenario.allow_clarifying_question and response.should_ask_clarifying_question:
            raise AssertionError("Агент задал уточняющий вопрос, хотя в сценарии это запрещено.")

    def check_clarifying_question(self, response: AgentResponse) -> None:
        """Проверяет, что агент задал уточняющий вопрос."""
        if not response.should_ask_clarifying_question:
            raise AssertionError("Ожидался уточняющий вопрос, но агент его не задал.")

        if not response.clarifying_question:
            raise AssertionError("Агент отметил уточняющий вопрос, но не вернул текст вопроса.")

    def check_recommendation(
            self,
            step: ScenarioStep,
            response: AgentResponse,
    ) -> None:
        """Проверяет, что агент выдал корректный список рекомендаций."""
        recommendations_count = len(response.recommended_cars)

        if recommendations_count < 1:
            raise AssertionError("На шаге рекомендации агент должен вернуть хотя бы одну машину.")

        if recommendations_count > 5:
            raise AssertionError("Агент не должен рекомендовать больше 5 машин.")

        if not step.acceptable_car_ids:
            return

        recommended_ids = [item.car_id for item in response.recommended_cars]

        if not self._has_intersection(recommended_ids, step.acceptable_car_ids):
            raise AssertionError(
                "Ни одна рекомендованная машина не входит в список допустимых машин сценария."
            )

    def check_selected_car(
            self,
            session: UserSession,
            expected_car_id: UUID,
    ) -> None:
        """Проверяет, что выбранная машина записана в сессию."""
        if session.selected_car_id != expected_car_id:
            raise AssertionError(
                f"Ожидалась выбранная машина {expected_car_id}, "
                f"но в сессии {session.selected_car_id}."
            )

        if session.dialog_status != DialogStatus.CAR_SELECTED:
            raise AssertionError("После выбора машины статус должен быть CAR_SELECTED.")

    def check_reservation_created(self, session: UserSession) -> None:
        """Проверяет, что mock-заявка создана."""
        if session.dialog_status != DialogStatus.RESERVATION_CREATED:
            raise AssertionError("Ожидался статус RESERVATION_CREATED.")

    def check_step_response(
            self,
            scenario: TestScenario,
            step: ScenarioStep,
            response: AgentResponse,
    ) -> None:
        """Проверяет ответ агента на конкретном шаге."""
        self.check_no_forbidden_clarification(scenario=scenario, response=response)

        if step.step_type == ScenarioStepType.EXPECT_CLARIFYING_QUESTION:
            self.check_clarifying_question(response=response)

        if step.step_type == ScenarioStepType.EXPECT_RECOMMENDATION:
            self.check_recommendation(step=step, response=response)

    def _has_intersection(
            self,
            left: List[UUID],
            right: List[UUID],
    ) -> bool:
        """Проверяет, пересекаются ли два списка UUID."""
        return bool(set(left).intersection(set(right)))
