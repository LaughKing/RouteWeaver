"""The four general-purpose roles, and the contract each one is held to."""

ROLES = ("planner", "executor", "verifier", "summarizer")
TERMINAL_ROLE = "summarizer"
NON_TERMINAL_ROLES = ("planner", "verifier")

_REASON = "Reason freely and show your work."

ROLE_BRIEF = {
    "planner": (
        "You are the PLANNER. Analyse the problem and break it into concrete "
        "sub-tasks that another worker could carry out. State what must be "
        "computed or proved, in what order, and what the difficulties are. "
        f"{_REASON} Do NOT solve the problem and do NOT state a final answer."),
    "executor": (
        "You are the EXECUTOR. Carry out the task you have been assigned. "
        f"{_REASON} Keep the full derivation. End with your candidate "
        "conclusion in the form <answer>your candidate answer</answer>."),
    "verifier": (
        "You are the VERIFIER. Check the output you have been given against the "
        "original problem. Do not solve the problem yourself and do not state a "
        "final answer.\n"
        "Reply in exactly one of these two forms:\n"
        "  PASS\n"
        "  FAIL\n"
        "  Location: <the specific step, line or claim that is wrong>\n"
        "  Reason: <why it is wrong>\n"
        "  Suggestion: <what to do instead>"),
    "summarizer": (
        "You are the SUMMARIZER. Reconcile the information you have been given, "
        "including any disagreement between candidates, and decide the answer. "
        "If candidates conflict, say briefly which you are taking and why. "
        f"{_REASON} End with <answer>your final answer</answer>."),
}

ROLE_MENU = {
    "planner": "analyses the problem and splits it into sub-tasks; never answers",
    "executor": "carries out one assigned task and returns a candidate answer",
    "verifier": "audits an existing output; returns PASS or FAIL with a reason; never answers",
    "summarizer": "reconciles everything and produces the final answer",
}


def build_worker_payload(role, raw_question, instruction, refs):
    """The exact text a worker receives."""
    parts = [ROLE_BRIEF[role], f"Original question:\n{raw_question.strip()}"]
    if refs:
        blocks = []
        for node_id, ref_role, text in refs:
            blocks.append(f"[{node_id}] (from {ref_role})\n{str(text).strip()}")
        parts.append("Earlier outputs you must use:\n\n" + "\n\n".join(blocks))
    if instruction and instruction.strip():
        parts.append(f"Your task:\n{instruction.strip()}")
    return "\n\n".join(parts)
