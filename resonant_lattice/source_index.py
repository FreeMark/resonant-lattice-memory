"""source_index.py - give extraction the URLs it never sees, and VERIFY the ones
it attaches.

THE PROBLEM. Extraction reads a transcript built from `episodes`, and episodes carry
only `user` and `assistant` rows. Tool results are not episodes, so the URLs a
research agent actually read never reach the extractor: measured on a live vet corpus,
112 episodes contained zero tool results, and 1,010 facts carried 71 refs -- the rest
null. The agent's own message was the ONLY channel a URL could arrive by, which is why
three successive prompt revisions failed to make provenance reliable. They were all
pushing on one narrow channel.

THE OBVIOUS FIX IS A TRAP, AND THIS MODULE EXISTS BECAUSE OF IT. Handing a model a
list of URLs and asking it to attach one per fact is an invitation to pick a
plausible-looking entry: a MENU, not evidence. On the same corpus a bag-of-words
matcher over full-page candidates produced 74 "recovered" refs of which ~67 were
false, and the tell was only visible in the shape of the result (72 fuzzy matches to
2 exact, a third of them landing off the source allowlist). A fabricated citation is
the one failure that would make a clinical corpus unsafe to hand to a professional.

So the feature has THREE parts, and the order below is the order that matters:

  ATTACH   find_ref() is the PRIMARY path and involves no model at all. It scans the
           pages the session actually retrieved and returns the one whose text
           verbatim contains the fact's quote. Measured 3 of 22 quoted facts on a
           live corpus -- low, but every one earned, against 0 for asking the model.

  VERIFY   verify_ref() re-checks any ref the model DID supply, to the same standard,
           and drops it if the page does not contain the quote. Defence in depth.

  SUPPLY   build_index_block() offers a `url | title` list in the extraction prompt.
           MEASURED INEFFECTIVE at 12B and default OFF: zero refs across 28 facts,
           including a real research block with 30 urls on offer. The one case that
           worked had urls inline beside each fact, where the model copied a
           100-character url correctly 3 times out of 3 -- so its problem was
           ASSOCIATION (matching a fact to one of thirty titles), not transcription.
           Citation markers ([1], [2], ...) would have made the copying cheaper and
           left the association exactly where it broke. Kept for larger extraction
           models; it costs ~1,200 prompt tokens and buys nothing at this size.

A ref that fails verification is DROPPED (set to None) and the fact is kept. Ties in
find_ref are refused the same way. That asymmetry is deliberate: losing a citation
costs recall, inventing one costs trust.

WHERE THE REMAINING CEILING IS, stated plainly so nobody hunts for a better matcher:
the extractor's source_quote comes from the agent's REPORT, and the agent paraphrases
the pages it read. A paraphrase is not present in any page, so no downstream rule can
recover its source. Raising 14% meaningfully means getting VERBATIM source text into
the report -- an upstream change, not a matching problem.

Pure functions only -- no DB, no LLM, no network -- so the rules above are unit
testable in isolation, which is how the false-match rate was found in the first place.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "SHINGLE_WORDS",
    "parse_tool_sources",
    "build_index_block",
    "find_ref",
    "verify_ref",
    "INDEX_HEADER",
]

# A run this long is a sentence fragment, not shared vocabulary. Tuned on the vet
# corpus: at 6 words the false-positive rate collapsed from ~90% to near zero.
SHINGLE_WORDS = 6

# Minimum body length for a retrieved page to count as CONTENT rather than a stub.
# Bot-mitigation shells on this corpus measured ~1.2KB of cookie banner; 500 chars
# keeps genuinely short pages while excluding empty ones.
MIN_BODY = 500

INDEX_HEADER = (
    "SOURCES RETRIEVED THIS SESSION (url | title). Use these for \"source_ref\". "
    "Attach a url ONLY to a fact whose source_quote you took from that page; a ref "
    "that does not match its page is removed automatically, so a guess is wasted "
    "work. Leave source_ref out when unsure."
)

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_URL_RE = re.compile(r'"url"\s*:\s*"([^"]+)"')
_DEC = json.JSONDecoder()

# Text that means "bot mitigation", not "page". A shell must never become the
# evidence a citation rests on.
_SHELL_MARKS = (
    "there doesn't seem to be anything here",
    "honeypot link", "access denied", "just a moment",
    "enable javascript", "are you a robot", "captcha", "403 forbidden",
)


def _norm(s: str) -> str:
    """Collapse whitespace and casefold. Punctuation is PRESERVED on purpose --
    stripping it would let '0.2-0.5' match a page containing '0205'."""
    return _WS.sub(" ", (s or "").replace(" ", " ")).strip().lower()


def _is_shell(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _SHELL_MARKS)


def _payload_items(content: str) -> List[Dict[str, Any]]:
    """Pull the result list out of one tool message, whatever shape it uses.

    Two shapes are produced by the web toolset, and both arrive wrapped in an
    <untrusted_tool_result> envelope with text AFTER the closing brace:

        extract -> {"results": [{url, title, content, error}]}
        search  -> {"success": true, "data": {"web": [{url, title, description}]}}

    raw_decode, NOT json.loads: the trailing envelope makes loads() raise on 100% of
    payloads, which silently turned an earlier version of this parser into a no-op.
    """
    if not content:
        return []
    i = content.find("{")
    if i < 0:
        return []
    try:
        parsed, _end = _DEC.raw_decode(content[i:])
    except Exception:                                            # noqa: BLE001
        # Unparseable (truncated storage). Salvage only when exactly one URL is
        # named -- a truncated multi-URL blob cannot say which URL the text came
        # from, and guessing there is the fabrication this module refuses to make.
        urls = set(_URL_RE.findall(content))
        if len(urls) == 1:
            return [{"url": next(iter(urls)), "title": "", "content": content}]
        return []
    if not isinstance(parsed, dict):
        return []
    if isinstance(parsed.get("results"), list):
        return [r for r in parsed["results"] if isinstance(r, dict)]
    data = parsed.get("data")
    if isinstance(data, dict) and isinstance(data.get("web"), list):
        return [r for r in data["web"] if isinstance(r, dict)]
    return []


def parse_tool_sources(messages: Optional[Iterable[Dict[str, Any]]],
                       tool_names: Iterable[str] = ("web_search", "web_extract"),
                       max_body: int = 20000) -> List[Dict[str, str]]:
    """-> [{"url", "title", "body"}] for every real page in `messages`.

    Deduplicated by url, first non-shell occurrence wins. Errors, shells and stubs
    are dropped here rather than downstream so a blocked fetch can never become the
    evidence behind a citation.
    """
    want = set(tool_names)
    out: List[Dict[str, str]] = []
    seen = set()
    for msg in (messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if msg.get("tool_name") and msg.get("tool_name") not in want:
            continue
        for r in _payload_items(msg.get("content") or ""):
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            if r.get("error"):
                continue
            title = (r.get("title") or "").strip()
            body = "%s %s" % (r.get("content") or "", r.get("description") or "")
            body = body.strip()
            if len(body) < MIN_BODY or _is_shell(title + " " + body):
                continue
            seen.add(url)
            out.append({"url": url, "title": title, "body": body[:max_body]})
    return out


def build_index_block(sources: List[Dict[str, str]], max_entries: int = 40,
                      title_chars: int = 90) -> str:
    """Render the compact index appended to the extraction transcript.

    Returns "" for an empty list so callers can concatenate unconditionally.

    THE CAPS ARE THE POINT. This block rides inside a prompt served at ctx 16384
    next to a ~2,000-token instruction preamble, and a research session can retrieve
    150+ distinct URLs. 40 entries at 90 title chars is roughly 1,200 tokens --
    enough for the model to tell an IRIS staging page from a stone-composition page,
    which is all the index has to do. Most-recent-first because research converges:
    the pages that answered the subtopic are the ones read last.
    """
    if not sources:
        return ""
    lines = [INDEX_HEADER]
    for s in list(reversed(sources))[:max_entries]:
        title = _WS.sub(" ", (s.get("title") or "")).strip()[:title_chars]
        lines.append("- %s | %s" % (s.get("url", ""), title))
    return "\n".join(lines)


def _longest_run(quote_words: List[str], page: str) -> int:
    """Longest run of consecutive quote words appearing verbatim in `page`."""
    n = len(quote_words)
    if n < SHINGLE_WORDS:
        return 0
    best = 0
    for i in range(n - SHINGLE_WORDS + 1):
        k = SHINGLE_WORDS
        while i + k <= n and " ".join(quote_words[i:i + k]) in page:
            best = max(best, k)
            k += 1
    return best


def find_ref(quote: str, bodies: Dict[str, str]) -> Optional[tuple]:
    """Find WHICH retrieved page a quote came from. -> (url, verdict, strength).

    THIS IS THE PRIMARY PATH, and it exists because asking the model to produce the
    url was measured to fail. Offering a `url | title` index in the extraction prompt
    produced ZERO refs across 28 facts, including a real research block with 30 urls
    on offer. The one case that worked had urls sitting inline beside each fact in the
    report, where the model copied a 100-character url correctly 3 times out of 3.

    So the model's problem was never TRANSCRIPTION, it was ASSOCIATION: matching a
    fact to one of thirty titles. Citation markers ([1], [2], ...) would have made the
    copying cheaper while leaving the association exactly where it broke. Removing the
    model from the loop is what actually fixes it.

    Running the check in reverse is strictly STRONGER than trusting a model-supplied
    ref, not merely cheaper: the model can only cite pages it happened to notice,
    while this scans every page the session retrieved. And it cannot fabricate --
    a url is returned only when the page verbatim contains the quote, the identical
    standard verify_ref applies.

    AMBIGUITY IS RESOLVED BY EVIDENCE, THEN REFUSED. The page with the longest
    verbatim overlap wins. If two pages tie at the top the quote genuinely does not
    identify a source (boilerplate repeated across a site, say), so None is returned
    and the fact keeps a null ref. Guessing between them is exactly the coin-flip
    citation this module exists to prevent.
    """
    if not quote or not bodies:
        return None
    q = _norm(quote)
    if not q:
        return None
    nums = set(_NUM.findall(q))
    qw = q.split()
    if len(qw) < SHINGLE_WORDS and not nums:
        return None

    # Cheap prefilter before the O(words x length) shingle scan: a page that lacks
    # one of the quote's numbers, or its longest word, cannot produce a verbatim run.
    # A research block stages 30-100 pages of up to 20k chars each, so skipping
    # hopeless candidates is what keeps this a few milliseconds per fact.
    longest = max(qw, key=len) if qw else ""
    scored = []
    for url, body in bodies.items():
        if not body:
            continue
        p = _norm(body)
        if nums and not all(x in p for x in nums):
            continue
        if len(longest) >= 6 and longest not in p:
            continue
        if q in p:
            scored.append((len(qw), url, "earned_exact"))
            continue
        run = _longest_run(qw, p)
        if run >= SHINGLE_WORDS:
            scored.append((run, url, "earned_shingle"))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        # Tie on strength: the quote does not identify one page. Refuse.
        return None
    run, url, verdict = scored[0]
    return (url, verdict, min(1.0, run / max(len(qw), 1)))


def verify_ref(quote: str, ref: str, bodies: Dict[str, str]) -> bool:
    """True when `ref` is a url from this session whose page really contains `quote`.

    Rules, in order, each one there because its absence produced a false citation on
    a real corpus:

      * the ref must be a url actually retrieved this session (membership in
        `bodies`) -- otherwise the model invented the url itself;
      * every NUMBER in the quote must appear in the page. No ratio, no tolerance:
        a misattributed reference interval is worse than no reference at all;
      * the quote must be an exact substring, OR share a run of >= SHINGLE_WORDS
        consecutive words with the page;
      * a quote too short to be distinctive (fewer than SHINGLE_WORDS words) is
        rejected unless it carries a number. "ammonium biurate crystalluria" is
        three words of standard terminology: matching it proves the page is about
        urinalysis, not that the claim came from it.
    """
    if not quote or not ref:
        return False
    body = bodies.get(ref)
    if not body:
        return False
    q, p = _norm(quote), _norm(body)
    if not q:
        return False
    nums = set(_NUM.findall(q))
    if nums and not all(x in p for x in nums):
        return False
    qw = q.split()
    if len(qw) < SHINGLE_WORDS and not nums:
        return False
    if q in p:
        return True
    return _longest_run(qw, p) >= SHINGLE_WORDS
