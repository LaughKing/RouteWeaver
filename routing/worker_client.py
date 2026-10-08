"""The worker side: one payload shape and one output constraint, in every mode.

THE OUTPUT CONSTRAINT LIVES HERE AND ONLY HERE.

An answer-format instruction belongs in the worker wrapper, never in the
question text. In the question it reaches a worker only in the modes whose
payload prepends the question -- single and multi do, an agentic role payload
does not -- so one line of dataset text can shorten worker replies in two
modes out of three and leave the third untouched, which is a difference between
modes that has nothing to do with routing. In the wrapper:

  * the ROUTER never sees it, so it cannot copy it into a sub-task;
  * every WORKER sees the same one, so no mode is advantaged by omission.

The constraint says where to PUT the answer, never how much the model may
think. `Reason freely` is load-bearing, not decoration.

PAYLOAD SHAPE. route_grammar builds two different worker payloads: multi gets
`Original question: ... Current sub-task: ...`, while an agentic role payload is
`Role: solver ... Assigned task: ...` with NO original question, so that worker
sees only the router's paraphrase. Here every call carries the verbatim
raw_question with the role-specific material appended under it, so a comparison
between modes is not also a comparison between payloads.
`include_question=False` reproduces the question-less agentic shape, and every
record stores which shape was used.
"""
import re

OUTPUT_CONSTRAINT = ("Reason freely and provide a complete derivation.\n"
                     "End with <answer>your final answer</answer>.")

# Guard against the original bug returning by any route: no worker payload may
# carry a brevity directive, whichever component assembled it.
BANNED_IN_PAYLOAD = [
    "final answer only", "answer only", "no explanation", "without explanation",
    "do not explain", "answer only yes or no", "yes or no", "one word",
    "short answer only", "confirm this", "is this correct", "be brief",
]
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


class PayloadContamination(AssertionError):
    """A brevity directive reached a worker from a part of the payload THIS CODE
    controls -- the constraint block or the question text. That is the original
    bug, it is ours, and a run must not continue past it."""


def check_payload(text):
    """Hard check, for the wrapper-controlled portion of a payload only."""
    low = str(text or "").lower()
    hits = [b for b in BANNED_IN_PAYLOAD if b in low]
    if hits:
        raise PayloadContamination(
            f"worker payload contains brevity directive(s) {hits} outside the "
            f"router's own sub-task; the dataset or this wrapper has "
            f"reintroduced the suppression bug")
    return text


def rubber_stamp_flags(task):
    """Brevity directives the ROUTER itself wrote into a sub-task.

    RECORDED, NOT RAISED, and the distinction decides the result. A router that
    writes "Verify: the distance is 4.5. Is this correct?" is exhibiting the
    behaviour under measurement; the sub-task is multi mode's real output.
    Killing the sample would drop precisely the trajectories where multi
    degenerates, so multi would be scored on its better half and look stronger
    than it is.

    A policy does produce "is this correct" occasionally DESPITE the prompt
    rule forbidding it, so this is a rate to report rather than a condition to
    assert away.
    """
    low = str(task or "").lower()
    return [b for b in BANNED_IN_PAYLOAD if b in low]


def build_worker_payload(raw_question, task=None, role=None, refs=None,
                         include_question=True):
    """The exact text a worker receives, identical in shape for every arm."""
    parts = [OUTPUT_CONSTRAINT]
    if role:
        parts.append(f"Role: {role}")
    if include_question:
        parts.append(f"Original question:\n{raw_question.strip()}")
    if refs:
        block = "\n\n".join(f"[{name}]\n{str(text).strip()}" for name, text in refs)
        parts.append(f"Referenced outputs:\n{block}")
    if task and task.strip() and task.strip() != raw_question.strip():
        parts.append(f"Current sub-task:\n{task.strip()}")
    return "\n\n".join(parts)


def extract_answer(text):
    r"""The <answer> block, else \boxed{...}, else the last meaningful line.

    THE \boxed FALLBACK IS NOT OPTIONAL. A maths model that skips the tag almost
    always closes with display math:

        ... Sum = 5+7+8+9+10 = 39.
        \[
        \boxed{39}
        \]

    Taking "the last non-empty line" there yields `\]`, scoring a correct reply
    as wrong. It does so unevenly, too: a router writes a proper <answer> block
    while a bare solver reply ends in the delimiter, so the loss falls on
    whichever path reads the worker's own text -- which can make a candidate
    oracle score BELOW plain single, an impossible-looking result that is
    entirely this function's doing.

    Bare delimiters are skipped rather than returned for the same reason.
    """
    found = _ANSWER_TAG.findall(str(text or ""))
    if found:
        return found[-1].strip(), "tag"
    boxed = _boxed_spans(str(text or ""))
    if boxed:
        return boxed[-1].strip(), "boxed"
    lines = [l.strip() for l in str(text or "").splitlines() if l.strip()]
    lines = [l for l in lines if not _DELIMITER_ONLY.match(l)]
    if not lines:
        return "", "empty"
    tail = lines[-1].strip().strip("*").strip()
    # prime_math parses "The answer is 39" but not "So 39" -- a leading
    # connective is enough to lose a correct answer, so strip the prose.
    tail = _PROSE_LEAD.sub("", tail).strip().rstrip(".").strip()
    return tail, "last_line"


_PROSE_LEAD = re.compile(
    r"^(?:so|thus|therefore|hence|then|and so|which gives|we get|this gives|"
    r"the\s+)?\s*(?:final\s+)?(?:answer|result|value)?\s*(?:is|:|=)?\s*",
    re.I)

# `\]`, `$$`, `---`, `\begin{...}` and friends carry no answer.
_DELIMITER_ONLY = re.compile(r"^(\\\]|\\\[|\$+|-{2,}|={2,}|\\begin\{[^}]*\}|"
                             r"\\end\{[^}]*\}|\**)$")


def _boxed_spans(text):
    r"""Every \boxed{...} body, brace-matched.

    A regex cannot do this: \boxed{\frac{9}{2}} has nested braces, and
    `\\boxed\{(.*?)\}` would return `\frac{9` .
    """
    out = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, start = 1, match.end()
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start:i - 1])
    return out


class WorkerPool:
    """Issues worker calls through ChatLLM, one client per worker id."""

    def __init__(self, dispatcher, max_tokens=2048, make_client=None):
        self.dispatcher = dispatcher
        self.max_tokens = max_tokens
        self._clients = {}
        self._make_client = make_client

    def _client(self, worker_id):
        if worker_id not in self._clients:
            if self._make_client is not None:
                self._clients[worker_id] = self._make_client(worker_id)
            else:
                from llm_client import ChatLLM
                self._clients[worker_id] = ChatLLM(
                    self.dispatcher, worker_id=worker_id,
                    default_max_tokens=self.max_tokens)
        return self._clients[worker_id]

    def call(self, worker_id, raw_question, task=None, role=None, refs=None,
             include_question=True, tag="", override_payload=None,
             max_tokens=None, exclude_texts=None):
        """One worker call.

        max_tokens is the pool's, NOT the channel entry's. The channel entries
        are sized for short answers; under `Reason freely and provide a
        complete derivation` the median reply is several times that, so falling
        back to a channel default silently clips the upper quartile of replies
        -- and it clips them unevenly across modes. The pool's cap is therefore
        asserted rather than trusted.
        """
        # override_payload lets a caller with its own role contract build the
        # message (roles.build_worker_payload does) while still going through
        # this pool's client, retry policy and record shape. The hygiene check
        # below still runs on it.
        payload = override_payload if override_payload is not None else \
            build_worker_payload(raw_question, task, role, refs, include_question)
        # The hard check covers only what THIS WRAPPER writes. Two exclusions,
        # both data rather than bugs: the router's own sub-task (recorded via
        # rubber_stamp_flags instead of raised), and the benchmark's question
        # text -- a LongBench source document natively contains "confirm this",
        # and killing a sample for its own reading material would delete rows
        # on content.
        # A THIRD exclusion, same category as the question text: the outputs of
        # EARLIER WORKERS, spliced in for a ref. A planner that ends its analysis
        # with "is this correct?" is data too -- it is not something this wrapper
        # wrote, and killing the run over another model's wording would end a
        # 34-hour baseline on a coin flip. Callers that pass override_payload pass
        # the referenced texts here; the constraint block, the role brief and the
        # rules text stay fully covered by the check.
        controlled = payload.replace(str(task or ""), "") \
                            .replace(str(raw_question or "").strip(), "")
        for chunk in (exclude_texts or []):
            chunk = str(chunk or "").strip()
            if chunk:
                controlled = controlled.replace(chunk, "")
        check_payload(controlled)
        flags = rubber_stamp_flags(task)
        record = {"worker_id": worker_id, "role": role, "tag": tag,
                  "payload_shape": "question+task" if include_question else "task_only",
                  "payload_chars": len(payload), "requested_task": task,
                  "rubber_stamp_flags": flags,
                  "router_wrote_rubber_stamp": bool(flags)}
        try:
            text, meta = self._client(worker_id).chat(
                [{"role": "user", "content": payload}], tag=tag,
                max_tokens=max_tokens, record_role="worker_turn")
        except Exception as exc:                      # noqa: BLE001
            # effective_worker_query is recorded on the FAILURE path too. It was
            # built before the call and is the only evidence of what the worker
            # would have been asked; omitting it made four payload-hygiene checks
            # read an empty string and report contamination where there was none.
            record.update({"success": False, "error": f"{type(exc).__name__}: {exc}",
                           "response_text": "", "output_tokens": 0,
                           "input_tokens": 0, "answer": "", "answer_source": "none",
                           "effective_worker_query": payload})
            return "", record
        answer, source = extract_answer(text)
        meta.pop("_text", None)
        record.update(meta)
        record.update({"success": True, "error": None, "response_text": text,
                       "answer": answer, "answer_source": source,
                       "effective_worker_query": payload})
        return text, record
