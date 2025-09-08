from typing import Dict, Any, List


def _format_recent_context(recent_context: List[Dict[str, Any]], limit: int = 3) -> str:
    if not recent_context:
        return ""
    context_str = "\nPrevious conversation context:\n"
    for ctx in recent_context[-limit:]:
        user = str(ctx.get("message", ""))
        assistant = str(ctx.get("response", ""))
        context_str += f"User: {user}\n"
        context_str += f"Assistant: {assistant}\n"
    return context_str


def build_agent_instruction_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    user_preferences: Dict[str, Any] | None,
    recent_context: List[Dict[str, Any]] | None,
) -> str:
    prefs = user_preferences or {}
    name = prefs.get("name", "")
    role = prefs.get("role", "")
    timezone = prefs.get("timezone", "UTC")
    work_hours = prefs.get("work_hours", "9:00-17:00")

    user_context = ""
    if name or role:
        user_context = "\nUser Information:\n"
        if name:
            user_context += f"- Name: {name}\n"
        if role:
            user_context += f"- Role: {role}\n"
        user_context += f"- Timezone: {timezone}\n"
        user_context += f"- Work Hours: {work_hours}\n"

    context_str = _format_recent_context(recent_context or [], limit=3)

    return (
        f"As {assistant_name}, the personal secretary for {company_name}, respond to this message from {sender}: \"{text}\"\n"
        f"{user_context}\n"
        f"{context_str}\n"
        "\n"
        "Remember to:\n"
        "- Be professional yet friendly\n"
        "- Provide clear, actionable information\n"
        "- Stay focused on administrative and office tasks\n"
        "- Maintain conversation context\n"
        "- Be proactive with scheduling and organization\n"
        "- Use any known user information appropriately\n"
        "- Consider user's timezone and work hours when discussing scheduling\n"
        "\n"
        "Response should reflect your role as the secretary for Sir Williams while being helpful and concise."
    )


def build_summary_prompt(contact_display: str, contact_phone: str, transcript: str) -> str:
    return (
        "You are an assistant that summarizes WhatsApp conversations for the owner. "
        "Given the recent transcript and contact info, produce a short summary (2-3 lines) "
        "and 3 concise agenda items or next steps. Format clearly.\n\n"
        f"Contact: {contact_display}, phone: {contact_phone}\n\n"
        f"Transcript:\n{transcript}\n\nSummary and Agenda:"
    )

