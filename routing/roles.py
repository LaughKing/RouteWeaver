"""The four general-purpose roles, and the contract each one is held to.

The question these four are designed around is whether a router given only
them, and no named topology, will assemble something useful. So there is no
MathSolver, no Coder, no "consensus" wiring: ensemble / refine / debate /
verified-consensus are shapes the router may or may not discover, never shapes
the code offers.

Each role's contract is enforced on the WORKER side, in the payload built here,
and checked afterwards in the trajectory record:

  planner     splits the problem. Must NOT answer it.
  executor    does an assigned task, keeps its full derivation, ends with a
              candidate answer.
  verifier    audits existing output. PASS, or FAIL plus where/why/how to fix.
              Must NOT answer.
  summarizer  reconciles what exists and emits the final answer.

Only summarizer is allowed to be the last node, so only summarizer is asked for
<answer>. A planner or verifier that emits one has broken its contract, and that
is recorded rather than silently accepted -- a verifier that answers is doing
the executor's job and the trajectory is no longer the topology it claims.
"""

ROLES = ("planner", "executor", "verifier", "summarizer")
TERMINAL_ROLE = "summarizer"
NON_TERMINAL_ROLES = ("planner", "verifier")

# Shared by every role: reasoning is never suppressed. A single "no
# explanation" line reaches only the modes whose payload carries the question,
# so it shortens worker replies unevenly across modes -- see worker_client.
# The only output instruction anywhere is about WHERE the answer goes.
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

# What the ROUTER is told about each role. Kept terse and behavioural: the router
# picks control flow, so it needs to know what a role is FOR, not how it is
# prompted.
ROLE_MENU = {
    "planner": "analyses the problem and splits it into sub-tasks; never answers",
    "executor": "carries out one assigned task and returns a candidate answer",
    "verifier": "audits an existing output; returns PASS or FAIL with a reason; never answers",
    "summarizer": "reconciles everything and produces the final answer",
}


def build_worker_payload(role, raw_question, instruction, refs):
    """The exact text a worker receives.

    EVERY worker gets the complete original question, whatever its role. The
    alternative -- passing only the router's paraphrase, as the agentic payload
    in route_grammar does -- means an agentic solver and a multi worker are
    never asked the same thing, so a comparison between the two modes partly
    measures the payload rather than the topology.

    `refs` is an ordered list of (node_id, role, text). Referenced outputs are
    spliced in by the environment, never copied forward by the router, so a
    verifier always sees the real upstream text rather than the router's summary
    of it.
    """
    parts = [ROLE_BRIEF[role], f"Original question:\n{raw_question.strip()}"]
    if refs:
        blocks = []
        for node_id, ref_role, text in refs:
            blocks.append(f"[{node_id}] (from {ref_role})\n{str(text).strip()}")
        parts.append("Earlier outputs you must use:\n\n" + "\n\n".join(blocks))
    if instruction and instruction.strip():
        parts.append(f"Your task:\n{instruction.strip()}")
    return "\n\n".join(parts)
