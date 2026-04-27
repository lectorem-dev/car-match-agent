import json
from typing import Any, Dict, Optional

from app.agents.extractor.extractor_prompts import EXTRACTOR_SYSTEM_PROMPT
from app.agents.extractor.extractor_schemas import ExtractorResult
from app.domain.user_session import DialogStatus, UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.session.session_update_service import SessionUpdateService
from app.utils.agent_logger import AgentLogColor, AgentLogger, detect_none_object_name


class Extractor:
    """Извлекает требования к машине из сообщения пользователя."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            session_update_service: SessionUpdateService,
            enable_logs: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.session_update_service = session_update_service
        self.logger = AgentLogger(
            "ExtractorAgent",
            enabled=enable_logs,
            color=AgentLogColor.BRIGHT_YELLOW,
        )

    def extract(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool,
            scenario_name: Optional[str] = None,
    ) -> ExtractorResult:
        """Извлекает требования из сообщения пользователя."""

        self.logger.start(
            scenario=scenario_name,
            user_message=user_message,
            dialog_status=session.dialog_status.value,
            allow_clarifying_question=allow_clarifying_question,
        )

        try:
            result = self._make_extraction(
                user_message=user_message,
                session=session,
                allow_clarifying_question=allow_clarifying_question,
            )

            self.logger.state(
                scenario=scenario_name,
                step="llm_result_parsed",
                has_result=result is not None,
                has_session_update=result.session_update is not None if result else False,
            )

            result = self._repair_if_needed(
                user_message=user_message,
                session=session,
                result=result,
                allow_clarifying_question=allow_clarifying_question,
            )

            self.logger.success(
                scenario=scenario_name,
                budget_max=result.session_update.budget_max,
                purpose=result.session_update.purpose,
                must_have=result.session_update.must_have,
                ask_question=result.should_ask_clarifying_question,
            )
            return result
        except Exception as error:
            self.logger.fail(
                error,
                scenario=scenario_name,
                user_message=user_message,
                has_current_session=session is not None,
                none_object=detect_none_object_name(
                    error,
                    current_session=session,
                ),
            )
            raise

    def _make_extraction(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool,
    ) -> ExtractorResult:
        """Делает первичное извлечение требований."""

        payload = {
            "user_message": user_message,
            "current_session": self._session_to_dict(session),
            "allow_clarifying_question": allow_clarifying_question,
        }

        raw_response = self.llm_client.generate(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=ExtractorResult.model_json_schema(),
            response_schema_name="extractor_result",
            max_output_tokens=1500,
        )

        return ExtractorResult.model_validate_json(raw_response)

    def _repair_if_needed(
            self,
            user_message: str,
            session: UserSession,
            result: ExtractorResult,
            allow_clarifying_question: bool,
    ) -> ExtractorResult:
        """Просит LLM исправить неполное извлечение требований."""

        temp_session = UserSession.model_validate(session.model_dump())

        temp_session = self.session_update_service.apply_update(
            session=temp_session,
            update=result.session_update,
        )

        if temp_session.has_required_data_for_recommendation():
            return result

        if allow_clarifying_question and session.dialog_status != DialogStatus.CLARIFYING_QUESTION:
            return result

        payload = {
            "user_message": user_message,
            "current_session": self._session_to_dict(session),
            "previous_result": result.model_dump(mode="json"),
            "problem": (
                "previous_result.session_update is incomplete or not useful enough. "
                "After applying it to current_session, required fields for recommendation are still missing. "
                "Required fields for recommendation: budget_max and purpose. "
                "Do not ask a clarifying question in this repair step. "
                "Return a FULL session_update object with all fields. "
                "Do not omit any SessionUpdate fields. "
                "Use null for unknown nullable fields, [] for unknown lists, and '' for empty user_notes. "
                "Extract all available requirements from user_message into session_update. "
                "If the user message contains a car usage scenario, purpose must be filled. "
                "If the user message is about cars but purpose is generic, use purpose='buy suitable car'."
            ),
            "required_output": (
                "Return corrected ExtractorResult. "
                "session_update must contain all SessionUpdate fields. "
                "Set should_ask_clarifying_question=false."
            ),
        }

        raw_response = self.llm_client.generate(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=ExtractorResult.model_json_schema(),
            response_schema_name="extractor_result_repair",
            max_output_tokens=1200,
        )

        return ExtractorResult.model_validate_json(raw_response)

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
