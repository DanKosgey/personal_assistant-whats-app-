from typing import Dict, Any, List, Optional


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
    tone: Optional[str] = None,
    language: Optional[str] = None,
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

    tone_hint = f"- Tone: {tone}\n" if tone else ""
    lang_hint = f"- Language: {language}\n" if language else ""

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
        f"{tone_hint}"
        f"{lang_hint}"
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


# Variant: scheduling
def build_scheduling_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    user_preferences: Dict[str, Any] | None,
    recent_context: List[Dict[str, Any]] | None,
    preferred_slots: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> str:
    prefs = user_preferences or {}
    tz = timezone or prefs.get("timezone", "UTC")
    context_str = _format_recent_context(recent_context or [], limit=4)
    slots = preferred_slots or prefs.get("preferred_slots") or []
    slots_str = "\n".join(f"- {s}" for s in slots) if slots else "- Propose 3 options within working hours"
    return (
        f"As {assistant_name} for {company_name}, handle a scheduling request from {sender}.\n"
        f"Message: \"{text}\"\n\n"
        f"Context:\n{context_str}\n"
        f"Guidelines:\n- Confirm timezone ({tz})\n- Offer mutually suitable times\n- Keep messages concise and polite\n- If unclear, ask one clarifying question\n\n"
        f"Suggested time slots:\n{slots_str}"
    )


# Variant: information retrieval
def build_information_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    knowledge_hints: Optional[List[str]] = None,
    recent_context: List[Dict[str, Any]] | None = None,
) -> str:
    hints = knowledge_hints or []
    context_str = _format_recent_context(recent_context or [], limit=3)
    hints_str = "\n".join(f"- {h}" for h in hints) if hints else "- Use existing context; avoid hallucinations"
    return (
        f"As {assistant_name} for {company_name}, provide accurate, concise information to {sender}.\n"
        f"Question: \"{text}\"\n\n"
        f"Relevant context:\n{context_str}\n"
        f"Constraints:\n- Cite assumptions or ask a brief clarifying question if needed\n"
        f"- Prefer bullet points when listing items\n"
        f"Guidance:\n{hints_str}"
    )


# Variant: task execution / action plan
def build_action_plan_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    recent_context: List[Dict[str, Any]] | None = None,
    max_steps: int = 5,
) -> str:
    context_str = _format_recent_context(recent_context or [], limit=3)
    return (
        f"As {assistant_name} for {company_name}, create a short action plan for {sender}.\n"
        f"Request: \"{text}\"\n\n"
        f"Context:\n{context_str}\n"
        f"Output:\n- Numbered steps (<= {max_steps})\n- Keep each step actionable and 1 sentence\n- Include any dependencies or prerequisites\n"
    )


# Variant: escalation to owner/manager
def build_escalation_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    owner_name: str,
    reason: Optional[str] = None,
    recent_context: List[Dict[str, Any]] | None = None,
) -> str:
    context_str = _format_recent_context(recent_context or [], limit=6)
    reason_line = f"Reason: {reason}\n" if reason else ""
    return (
        f"Draft a concise escalation note from {assistant_name} to {owner_name}.\n"
        f"Incoming message from {sender}: \"{text}\"\n"
        f"{reason_line}\n"
        f"Recent context:\n{context_str}\n"
        f"Include:\n- Short summary (<=2 sentences)\n- 3 bullet next steps\n- Any blockers or deadlines\n"
    )


