from typing import Dict, Any, List, Optional, Union


def _format_recent_context(recent_context: List[Dict[str, Any]], limit: int = 3) -> str:
    if not recent_context:
        return ""
    
    context_str = "\n## Recent Conversation History\n"
    for i, ctx in enumerate(recent_context[-limit:], 1):
        user = str(ctx.get("message", "")).strip()
        assistant = str(ctx.get("response", "")).strip()
        if user or assistant:
            context_str += f"\n**Exchange {i}:**\n"
            if user:
                context_str += f"👤 User: {user}\n"
            if assistant:
                context_str += f"🤖 Assistant: {assistant}\n"
    return context_str


def build_agent_instruction_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    user_preferences: Optional[Dict[str, Any]],
    recent_context: Optional[List[Dict[str, Any]]],
    tone: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    prefs = user_preferences or {}
    name = prefs.get("name", "").strip()
    role = prefs.get("role", "").strip()
    timezone = prefs.get("timezone", "UTC").strip()
    work_hours = prefs.get("work_hours", "9:00-17:00").strip()

    user_context = ""
    if name or role:
        user_context = "\n## User Profile\n"
        if name:
            user_context += f"• **Name:** {name}\n"
        if role:
            user_context += f"• **Role:** {role}\n"
        user_context += f"• **Timezone:** {timezone}\n"
        user_context += f"• **Work Hours:** {work_hours}\n"

    context_str = _format_recent_context(recent_context or [])

    tone_section = f"• **Tone:** {tone}\n" if tone else ""
    lang_section = f"• **Language:** {language}\n" if language else ""

    return f"""# Role: {assistant_name}
## Company: {company_name}
## Task: Respond to message from {sender}

### Message Content:
"{text}"

{user_context}
{context_str}

## Response Guidelines:
1. **Professional Excellence**:
   - Maintain polished corporate communication standards
   - Balance friendliness with professional efficiency
   - Use appropriate business etiquette

2. **Context Awareness**:
   - Leverage conversation history for continuity
   - Incorporate user preferences naturally
   - Consider timezone ({timezone}) and work hours ({work_hours}) for scheduling

3. **Communication Standards**:
{tone_section}{lang_section}   - Use active voice and clear phrasing
   - Keep responses concise but thorough
   - Structure complex information with bullet points when helpful

4. **Personalization**:
   - Address user by name when known
   - Reference previous interactions when relevant
   - Adapt to user's communication style

5. **Critical Thinking**:
   - Read between lines for unstated needs
   - Prioritize urgency and importance appropriately
   - Suggest proactive solutions when possible

6. **Scope Management**:
   - Focus on administrative and organizational tasks
   - Escalate appropriately when needed
   - Maintain appropriate boundaries

## Response Format:
- Use natural, conversational language
- Include appropriate greeting/closing
- Structure information for clarity
- Emphasize key points tactfully
- Provide clear next steps when applicable"""


def build_summary_prompt(contact_display: str, contact_phone: str, transcript: str) -> str:
    return f"""# Task: Conversation Summary
## Contact: {contact_display} ({contact_phone})

## Transcript:
{transcript}

## Analysis Framework:
1. **Purpose Identification**: Determine primary objective of interaction
2. **Key Information Extraction**: Identify critical data points exchanged
3. **Action Tracking**: Note commitments and decisions made
4. **Urgency Assessment**: Evaluate time sensitivity (🔴 High | 🟡 Medium | 🟢 Low)
5. **Context Evaluation**: Consider relationship and historical context

## Output Format:
**TL;DR:** [Single-sentence essence of conversation]

**Highlights:**
• [Most significant point 1]
• [Most significant point 2]
• [Additional important information]

**Next Actions:**
• [Immediate required action]
• [Follow-up needed]
• [Long-term consideration]

**Priority Level:** [🔴|🟡|🟢] [Rationale for priority rating]

**Context Notes:**
[Relevant background information or special considerations]"""


def build_short_summary_prompt(contact_display: str, contact_phone: str, transcript: str) -> str:
    return f"""# Task: Quick Summary
## Contact: {contact_display} ({contact_phone})

## Transcript:
{transcript}

## Requirements:
- Extract core message essence
- Determine urgency level (🔴 🟡 🟢)
- Identify any immediate action required
- Maximum 2-line summary

## Output Format:
[🔴|🟡|🟢] [Concise summary of key message]
[Action: Specific next step if applicable]"""


def build_detailed_summary_prompt(contact_display: str, contact_phone: str, transcript: str) -> str:
    return f"""# Task: Comprehensive Analysis
## Contact: {contact_display} ({contact_phone})

## Transcript:
{transcript}

## Analysis Dimensions:
1. **Conversation Flow**: Chronological development of discussion
2. **Key Themes**: Main topics and subjects covered
3. **Decision Points**: Agreements, disagreements, and conclusions
4. **Action Items**: Specific commitments made
5. **Contextual Factors**: External circumstances affecting conversation
6. **Sentiment Analysis**: Emotional tone and undertones

## Output Format:
**Executive Summary:**
[One-paragraph comprehensive overview]

**Timeline Analysis:**
• [Key event 1] → [Key event 2] → [Key event 3]

**Key Points:**
• Theme 1: [Explanation]
• Theme 2: [Explanation]
• Theme 3: [Explanation]

**Action Plan:**
1. [Immediate action] (Priority: 🔴)
2. [Follow-up action] (Priority: 🟡)
3. [Monitoring item] (Priority: 🟢)

**Contextual Notes:**
[Relevant background information and implications]

**Risk Assessment:**
[Potential challenges or considerations]"""


def build_scheduling_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    user_preferences: Optional[Dict[str, Any]],
    recent_context: Optional[List[Dict[str, Any]]],
    preferred_slots: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> str:
    prefs = user_preferences or {}
    tz = timezone or prefs.get("timezone", "UTC")
    context_str = _format_recent_context(recent_context or [])
    slots = preferred_slots or prefs.get("preferred_slots") or []
    
    slots_str = "\n".join(f"• {s}" for s in slots) if slots else "• Propose 3 optimal time windows within standard business hours"

    return f"""# Role: {assistant_name}
## Company: {company_name}
## Task: Schedule Meeting Request

### From: {sender}
### Message: "{text}"

{context_str}

## Scheduling Parameters:
• **Timezone:** {tz}
• **Preferred Slots:**
{slots_str}

## Guidelines:
1. **Availability Management**:
   - Avoid scheduling outside work hours ({prefs.get('work_hours', '9:00-17:00')})
   - Buffer between meetings (minimum 15 minutes)
   - Consider travel time if applicable

2. **Communication Protocol**:
   - Confirm timezone awareness
   - Offer 2-3 options when possible
   - Specify meeting duration clearly
   - Include video/conference details if relevant

3. **Contingency Planning**:
   - Suggest alternative formats if scheduling fails
   - Propose delegate availability when appropriate
   - Escalate complex scheduling needs

4. **Professional Standards**:
   - Use timezone-aware terminology
   - Confirm all details in writing
   - Provide calendar integration options"""


def build_information_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    knowledge_hints: Optional[List[str]] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    hints = knowledge_hints or []
    context_str = _format_recent_context(recent_context or [])
    
    hints_str = "\n".join(f"• {h}" for h in hints) if hints else "• Use available knowledge base\n• Acknowledge information limitations"

    return f"""# Role: {assistant_name}
## Company: {company_name}
## Task: Information Response

### Query from {sender}:
"{text}"

{context_str}

## Knowledge Base:
{hints_str}

## Response Protocol:
1. **Accuracy Verification**:
   - Cross-reference multiple sources when possible
   - Qualify uncertainty when information incomplete
   - Provide creation/last-updated dates for time-sensitive info

2. **Structuring Principles**:
   - Lead with key information
   - Use bullet points for complex information
   - Group related concepts together
   - Highlight critical details

3. **Scope Management**:
   - Stay within organizational knowledge boundaries
   - Redirect to appropriate channels when needed
   - Protect confidential information

4. **Follow-up Planning**:
   - Suggest additional resources
   - Offer to deep-dive if needed
   - Provide contact points for specialized questions"""


def build_action_plan_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    company_name: str,
    recent_context: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 5,
) -> str:
    context_str = _format_recent_context(recent_context or [])
    
    return f"""# Role: {assistant_name}
## Company: {company_name}
## Task: Action Plan Development

### Request from {sender}:
"{text}"

{context_str}

## Action Plan Requirements:
• Maximum {max_steps} clear, executable steps
• Logical sequence and dependencies
• Realistic timeframe estimates
• Resource requirements identification
• Success criteria definition

## Output Format:
**Objective:** [Clear statement of goal]

**Steps:**
1. [Actionable step 1] (Duration: [estimate])
   → Prerequisites: [requirements]
   → Output: [expected result]

2. [Actionable step 2] (Duration: [estimate])
   → Prerequisites: [requirements]
   → Output: [expected result]

**Success Metrics:**
- [Measurable metric 1]
- [Measurable metric 2]

**Risks:**
- [Potential challenge 1] → [Mitigation strategy]
- [Potential challenge 2] → [Mitigation strategy]"""


def build_escalation_prompt(
    sender: str,
    text: str,
    assistant_name: str,
    owner_name: str,
    reason: Optional[str] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    context_str = _format_recent_context(recent_context or [], limit=6)
    reason_line = f"\n**Escalation Reason:** {reason}" if reason else ""

    return f"""# Task: Management Escalation
## From: {assistant_name}
## To: {owner_name}
## Regarding: Message from {sender}

### Original Message:
"{text}"
{reason_line}

{context_str}

## Escalation Framework:
**Urgency Level:** [🔴 Critical | 🟡 Elevated | 🟢 Informational]

**Summary:**
[Brief situation description including why this requires escalation]

**Background Context:**
[Relevant historical information and previous interactions]

**Immediate Concerns:**
• [Primary issue 1]
• [Primary issue 2]

**Potential Impact:**
[Business, financial, or operational consequences]

**Recommended Actions:**
1. [Immediate next step]
2. [Stakeholder to involve]
3. [Timeline consideration]

**Information Gaps:**
[Missing information that would be helpful for resolution]"""