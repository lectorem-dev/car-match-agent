from app.agents.extractor.schemas import SessionUpdate
from app.domain.user_session import DialogStatus, UserSession

class SessionUpdateService:
    """Сервис для применения SessionUpdate к UserSession.    """

    def apply_update(self, session, update):
        pass

    def apply_update(self, session, update):
        pass


def _merge_notes(
        old_notes: str,
        new_notes: str,
) -> str:
    """Склеивает старые и новые заметки пользователя."""

    if not old_notes:
        return new_notes.strip()

    return f"{old_notes.strip()} {new_notes.strip()}"


def _refresh_dialog_status(session: UserSession) -> None:
    """Обновляет статус диалога после изменения памяти."""

    # Если уже выбрана машина или создана заявка, не откатываем статус назад.
    if session.dialog_status in {
        DialogStatus.CAR_SELECTED,
        DialogStatus.READY_FOR_RESERVATION,
        DialogStatus.RESERVATION_CREATED,
    }:
        return

    # Если собран минимум данных, можно переходить к рекомендациям.
    if session.has_required_data_for_recommendation():
        session.dialog_status = DialogStatus.READY_TO_RECOMMEND
    else:
        session.dialog_status = DialogStatus.INITIAL_SURVEY


def _extend_unique(
        target: list[str],
        values: list[str],
) -> None:
    """Добавляет новые значения в список без дублей."""

    existing_values = {value.lower() for value in target}

    for value in values:
        normalized_value = value.strip()

        if not normalized_value:
            continue

        if normalized_value.lower() in existing_values:
            continue

        target.append(normalized_value)
        existing_values.add(normalized_value.lower())


def apply_update(
        session: UserSession,
        update: SessionUpdate,
) -> UserSession:
    """Обновляет память диалога данными, которые извлек агент."""

    if update.budget_min is not None:
        session.budget_min = update.budget_min

    if update.budget_max is not None:
        session.budget_max = update.budget_max

    if update.purpose is not None:
        session.purpose = update.purpose

    if update.experience_level is not None:
        session.experience_level = update.experience_level

    if update.family_size is not None:
        session.family_size = update.family_size

    _extend_unique(session.preferred_body_types, update.preferred_body_types)
    _extend_unique(session.preferred_brands, update.preferred_brands)

    _extend_unique(session.must_have, update.must_have)
    _extend_unique(session.must_not_have, update.must_not_have)

    if update.user_notes:
        session.user_notes = _merge_notes(
            old_notes=session.user_notes,
            new_notes=update.user_notes,
        )

    _refresh_dialog_status(session)

    return session