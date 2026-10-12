"""Token positions of the worker id inside a policy-generated turn."""
import re

_MODEL_ATTR = re.compile(r'model="([^"]*)"')


def token_char_offsets(tokenizer, token_ids):
    """[(start, end)] in the decoded text, one per token."""
    offs = []
    pos = 0
    for tid in token_ids:
        piece = tokenizer.decode([tid])
        offs.append((pos, pos + len(piece)))
        pos += len(piece)
    return offs, pos


def worker_spans(tokenizer, token_ids, allowed=None):
    """-> [(tok_start, tok_end, recovered_id)] for each model="ID" in the turn."""
    if not token_ids:
        return []
    offs, _ = token_char_offsets(tokenizer, token_ids)
    text = "".join(tokenizer.decode([t]) for t in token_ids)
    out = []
    for m in _MODEL_ATTR.finditer(text):
        wid = m.group(1)
        if allowed is not None and wid not in allowed:
            continue
        a, b = m.start(1), m.end(1)
        idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a]
        if idx:
            out.append((idx[0], idx[-1] + 1, wid))
    return out
