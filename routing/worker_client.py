"""The worker side: one payload shape and one output constraint, in every mode."""
import re

OUTPUT_CONSTRAINT = ("Reason freely and provide a complete derivation.\n"
                     "End with <answer>your final answer</answer>.")

BANNED_IN_PAYLOAD = [
    "final answer only", "answer only", "no explanation", "without explanation",
    "do not explain", "answer only yes or no", "yes or no", "one word",
    "short answer only", "confirm this", "is this correct", "be brief",
]
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


class PayloadContamination(AssertionError):
    """A brevity directive appeared in the wrapper-controlled part of a payload."""


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
    """Brevity directives the ROUTER itself wrote into a sub-task."""
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
    r"""The <answer> block, else the last boxed answer, else the last meaningful line."""
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
    tail = _PROSE_LEAD.sub("", tail).strip().rstrip(".").strip()
    return tail, "last_line"


_PROSE_LEAD = re.compile(
    r"^(?:so|thus|therefore|hence|then|and so|which gives|we get|this gives|"
    r"the\s+)?\s*(?:final\s+)?(?:answer|result|value)?\s*(?:is|:|=)?\s*",
    re.I)

_DELIMITER_ONLY = re.compile(r"^(\\\]|\\\[|\$+|-{2,}|={2,}|\\begin\{[^}]*\}|"
                             r"\\end\{[^}]*\}|\**)$")


def _boxed_spans(text):
    r"""Every boxed answer body, with nested braces matched."""
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
        """One worker call."""
        payload = override_payload if override_payload is not None else \
            build_worker_payload(raw_question, task, role, refs, include_question)
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
