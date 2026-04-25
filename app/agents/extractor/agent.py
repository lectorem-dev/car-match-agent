import json
from typing import Any, Dict

from app.agents.extractor.prompts import EXTRACTOR_SYSTEM_PROMPT
from app.agents.extractor.schemas import ExtractorResult
from app.domain.user_session import DialogStatus, UserSession
from app.llm.yandex_llm_client import YandexLLMClient
from app.services.session_update_service import SessionUpdateService


class Extractor:
    """Извлекает требования к машине из сообщения пользователя."""

    def __init__(
            self,
            llm_client: YandexLLMClient,
            session_update_service: SessionUpdateService,
    ) -> None:
        self.llm_client = llm_client
        self.session_update_service = session_update_service

    def extract(
            self,
            user_message: str,
            session: UserSession,
            allow_clarifying_question: bool,
    ) -> ExtractorResult:
        """Извлекает требования из сообщения пользователя."""

        result = self._make_extraction(
            user_message=user_message,
            session=session,
            allow_clarifying_question=allow_clarifying_question,
        )

        result = self._repair_if_needed(
            user_message=user_message,
            session=session,
            result=result,
            allow_clarifying_question=allow_clarifying_question,
        )

        return result

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
                "previous_result.session_update is incomplete. "
                "After applying it to current_session, required fields for recommendation are still missing. "
                "Required fields: budget_max and purpose. "
                "Do not ask a clarifying question in this repair step. "
                "Extract all available requirements from user_message into session_update. "
                "If the user message contains a car usage scenario, purpose must be filled. "
                "If the user message is about cars but purpose is generic, use purpose='buy suitable car'."
            ),
            "required_output": (
                "Return corrected ExtractorResult. "
                "Set should_ask_clarifying_question=false."
            ),
        }

        raw_response = self.llm_client.generate(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            response_schema=ExtractorResult.model_json_schema(),
            response_schema_name="extractor_result_repair",
        )

        return ExtractorResult.model_validate_json(raw_response)

    def _session_to_dict(self, session: UserSession) -> Dict[str, Any]:
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