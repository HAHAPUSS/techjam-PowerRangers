"""Conversational shopping agent for the TechJam e-commerce search challenge.

Architecture
------------
1. **Offline catalog index** (built once, standard library only): every product is
   flattened into a normalized text blob, bucketed by its coarse category, and
   given a popularity prior derived from its rating volume and mean rating.
2. **Dialogue understanding**: each customer turn is parsed into structured
   constraints (category, material, colour, budget, free-text requirements) that
   accumulate in per-session state, including retraction handling when the
   customer overrides an earlier preference.
3. **Retrieval**: the stated category narrows the candidate pool, then candidates
   are ranked by a blend of constraint evidence (exact phrase evidence first,
   then token overlap) and the popularity prior.
4. **Clarification policy**: the agent always returns its current best ranking and
   simultaneously asks the question with the highest expected information gain,
   so no turn is spent on retrieval or on questioning alone.

Optional LLM layer (OpenAI or xAI / Grok)
-----------------------------------------
`starter/llm_client.py` adds an *optional* model backend, enabled only when
``OPENAI_API_KEY`` or ``XAI_API_KEY`` plus the matching model id is set. It is
wired at one point: when deterministic parsing fails to recognise a customer
utterance, the model is asked to extract the structure instead. On the literal
protocol that never happens, so the agent makes **zero** API calls and reports
zero tokens; the model exists to absorb paraphrased or free-form customers. An
optional semantic reranker (``AGENT_LLM_RERANK=1``) is off by default - measured,
it costs more than it returns. Every LLM path falls back to the deterministic
result on any error, so the agent runs unchanged with no key, no network, or a
failing endpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

try:  # optional: absent or unconfigured means the deterministic path
    from starter.llm_client import build_default_client
except Exception:  # pragma: no cover - the agent must load either way
    def build_default_client():  # type: ignore[misc]
        return None


TOKEN_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")

MAX_RECOMMENDATIONS = 100

# Attribute vocabulary accepted by the `ask_attribute` field of the contract.
ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "suede", "satin", "cashmere",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "still", "exploring", "not", "quite", "right", "yet", "have",
    "dont", "prefer", "preference", "need", "need", "actually", "ignore",
    "earlier", "what", "matters", "key", "requirement", "about", "one",
    "specific", "attribute", "ask", "judgment", "use", "your", "additional",
}

# Generic category labels that carry no discriminative signal.
GENERIC_CATEGORY_LABELS = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}

# --- Customer utterance patterns -------------------------------------------
RE_OPEN_BUYING = re.compile(r"^i'm looking for (.+?)\. a key requirement is: (.+?)\.?$", re.I)
RE_OPEN_BROWSING = re.compile(r"^i'm looking for (.+?), but i'm still exploring\.?$", re.I)
RE_OPEN_GENERIC = re.compile(r"^i'm looking for (.+?)\. (.+?)\.?$", re.I)
RE_OPEN_BARE = re.compile(r"^i'm looking for (.+?)\.?$", re.I)
RE_REVEAL = re.compile(r"what matters is:\s*(.+?)\.?$", re.I)
RE_OVERRIDE = re.compile(r"ignore my earlier preference\.\s*what i need is:\s*(.+?)\.?$", re.I)
RE_NOTHING_MORE = re.compile(r"(?:don't|do not) have an additional preference for ([a-z_]+)", re.I)
RE_NO_OPINION = re.compile(r"(?:don't|do not) have a(?:ny)? (?:strong )?preference (?:for|on|about) ([a-z_]+)", re.I)
RE_COLOR_CONSTRAINT = re.compile(r"^colou?r:\s*(.+)$", re.I)
RE_BUDGET_CONSTRAINT = re.compile(r"budget around \$?([0-9]+(?:\.[0-9]+)?)", re.I)
RE_PRICE_IN_TEXT = re.compile(r"(?:\$|under |below |around |about )\s*([0-9]+(?:\.[0-9]+)?)", re.I)

# Natural-language phrasing for each attribute we may probe.
QUESTION_TEXT = {
    "other": "Got it. Is there anything else that matters to you — material, colour, fit, or budget?",
    "material": "What material are you hoping for?",
    "color": "Do you have a colour in mind?",
    "style": "What style or fit works best for you?",
    "use_case": "What will you mainly be using it for?",
    "budget": "Roughly what budget are you working with?",
    "size": "What size or fit do you need?",
    "brand": "Is there a brand you prefer or want to avoid?",
    "feature": "Any specific features you need it to have?",
    "category": "Which type of item are you after exactly?",
}

# Order in which specific attributes are probed once the open question is spent.
PROBE_ORDER = ("material", "color", "style", "use_case", "budget", "size", "feature", "brand")

# --- Scoring weights (tuned on the public development set) -----------------
W_PHRASE_BASE = 4.0          # credit for an exactly quoted requirement
W_PHRASE_PER_TOKEN = 0.60    # longer quoted requirements are more discriminative
W_PHRASE_TOKEN_CAP = 12
W_OVERLAP_STRONG = 4.0       # partial (token-level) requirement match
W_OVERLAP_WEAK = 1.5
W_ATTRIBUTE_HIT = 1.5        # material / colour mentioned by the customer
W_BUDGET_CLOSE = 3.0
W_BUDGET_NEAR = 1.5
W_BUDGET_OFF = -0.75
W_CATEGORY_EXACT = 8.0       # only used on the whole-catalog fallback path
W_CATEGORY_OVERLAP = 3.0
W_POPULARITY = 0.60          # log rating volume: a reliable purchase-likelihood prior
W_RATING = 0.25
W_SOFT_TOKEN = 0.05          # aggregate-profile preference tags: tie-break only
IDF_FLOOR = 1.00             # matched requirements keep full strength ...
IDF_SPAN = 6.00              # ... amplified when rarely satisfied across the pool
RETRACTED_WEIGHT = 0.45      # overridden preferences stay as a weak prior
FREE_TEXT_WEIGHT = 0.70      # requirements recovered from unstructured phrasing
FUZZY_CATEGORY_FLOOR = 0.60  # minimum token overlap to accept a fuzzy product type
LLM_WEIGHT = 0.90            # requirements recovered by the optional LLM extractor
LLM_RERANK_DEPTH = 20        # how many candidates the optional reranker may reorder

EXTRACT_SYSTEM = (
    "You extract shopping requirements from one customer message in a product-search chat. "
    "Reply with a single JSON object and no prose, using exactly these keys: "
    '{"product_type": string|null, "requirements": [string], '
    '"retracts_earlier": boolean, "nothing_more_to_add": boolean}. '
    "product_type is the kind of item wanted (e.g. \"running shoes\", \"pendant necklace\"), "
    "or null if this message does not say. "
    "requirements are the attributes the customer states - material, colour, size, style, "
    "use case, budget, features. "
    "CRITICAL: write each requirement the way it would appear in a product listing, not as "
    "conversational prose. Strip filler verbs and articles. Keep each to 1-5 words. "
    'Say "leather" not "made of leather"; "budget around $80" not "priced under $80"; '
    '"machine wash" not "it should be machine washable". Use [] if none are stated. '
    "retracts_earlier is true only if they withdraw or replace something said earlier. "
    "nothing_more_to_add is true only if they say they have no further preferences."
)

RERANK_SYSTEM = (
    "You re-rank candidate products against a customer's stated requirements. "
    'Reply with a single JSON object: {"order": [numbers]} listing every candidate number '
    "exactly once, best match first. No prose."
)
EVIDENCE_SATURATION_MIN_CONSTRAINTS = 4


def _normalize(text: str) -> str:
    return WS_RE.sub(" ", text).strip().lower()


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) > 1 and token not in STOPWORDS}


def _flatten(value: object) -> list[str]:
    """Render a catalog field as the list of strings a customer could quote."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value != "" else []


def _coarse_category(values: list[str]) -> str:
    """The customer-facing product type, e.g. 'Shoes Fashion Sneakers'."""
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_CATEGORY_LABELS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _as_price(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", value)
        if match:
            price = float(match.group(0))
            return price if price > 0 else None
    return None


class Constraint:
    """One requirement stated by the customer, with how strongly it binds."""

    __slots__ = ("kind", "text", "tokens", "value", "weight")

    def __init__(self, kind: str, text: str, weight: float = 1.0, value: float | None = None) -> None:
        self.kind = kind
        self.text = text
        self.weight = weight
        self.value = value
        self.tokens = _content_tokens(text)

    def demote(self, weight: float) -> None:
        self.weight = min(self.weight, weight)


class SessionState:
    __slots__ = ("category", "category_tokens", "constraints", "seen", "asked",
                 "open_question_exhausted", "free_tokens", "info_exhausted", "bucket")

    def __init__(self) -> None:
        self.category: str | None = None
        self.category_tokens: set[str] = set()
        self.bucket: list[int] | None = None
        self.constraints: list[Constraint] = []
        self.seen: set[str] = set()
        self.asked: set[str] = set()
        self.open_question_exhausted = False
        self.info_exhausted = False
        self.free_tokens: set[str] = set()

    def add(
        self,
        kind: str,
        text: str,
        weight: float = 1.0,
        value: float | None = None,
        promote_existing: bool = False,
    ) -> None:
        key = f"{kind}\0{_normalize(text)}"
        if not text.strip():
            return
        if key in self.seen:
            if promote_existing:
                for constraint in self.constraints:
                    if constraint.kind == kind and _normalize(constraint.text) == _normalize(text):
                        constraint.weight = max(constraint.weight, weight)
                        break
            return
        self.seen.add(key)
        self.constraints.append(Constraint(kind, text, weight, value))


class Agent:
    """Multi-turn retrieval agent: parse, accumulate constraints, rank, clarify."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", llm: object | None = None) -> None:
        # The LLM is strictly optional. With no key, no network, or any error
        # at all, `self.llm` is None and the agent runs fully deterministically.
        self.llm = llm if llm is not None else build_default_client()
        self._llm_rerank = os.environ.get("AGENT_LLM_RERANK", "0") == "1"
        self.catalog_path = Path(catalog_path)
        self.asins: list[str] = []
        self.blobs: list[str] = []          # normalized title + features + details + categories
        self.titles: list[str] = []
        self.prices: list[float | None] = []
        self.priors: list[float] = []
        self.coarse: list[str] = []
        self.buckets: dict[str, list[int]] = {}
        self._bucket_tokens: dict[str, set[str]] = {}
        self._token_cache: dict[int, set[str]] = {}
        self._sessions: dict[str, SessionState] = {}
        self._build_index()

    # -- indexing -----------------------------------------------------------
    def _build_index(self) -> None:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                product = json.loads(line)
                categories = [str(value) for value in (product.get("categories") or [])]
                coarse = _coarse_category(categories)
                parts = [str(product.get("title") or "")]
                parts.extend(_flatten(product.get("features")))
                parts.extend(_flatten(product.get("details")))
                parts.extend(categories)
                store = product.get("store")
                if store:
                    parts.append(str(store))
                self.asins.append(str(product["parent_asin"]))
                self.titles.append(str(product.get("title") or ""))
                self.blobs.append(_normalize(" | ".join(parts)))
                self.prices.append(_as_price(product.get("price")))
                self.coarse.append(coarse)
                rating_number = product.get("rating_number") or 0
                average_rating = product.get("average_rating") or 0.0
                self.priors.append(
                    W_POPULARITY * math.log1p(float(rating_number))
                    + W_RATING * (float(average_rating) - 3.5)
                )
                self.buckets.setdefault(coarse.lower(), []).append(index)
        self._bucket_tokens = {name: _content_tokens(name) for name in self.buckets}

    def _blob_tokens(self, index: int) -> set[str]:
        tokens = self._token_cache.get(index)
        if tokens is None:
            tokens = _content_tokens(self.blobs[index])
            self._token_cache[index] = tokens
        return tokens

    def llm_stats(self) -> dict:
        """Model usage for the submission's cost/latency disclosure."""
        if self.llm is None:
            return {"enabled": False, "reason": "no LLM configured; deterministic path only"}
        stats = self.llm.stats() if hasattr(self.llm, "stats") else {}
        return {"enabled": True, **stats}

    # -- contract -----------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState()
        self._sessions[session_id] = state
        self._token_cache.clear()
        # The profile is an anonymized aggregate; its preference tags are used
        # only as a light tie-break, never as a hard filter.
        if isinstance(user_profile, dict):
            for tag in user_profile.get("preference_tags") or []:
                state.free_tokens.update(_content_tokens(str(tag)))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:                      # defensive: never raise inside a session
            state = SessionState()
            self._sessions[session_id] = state
        before = self.llm.usage_snapshot() if self.llm is not None else (0, 0)
        if not self._observe(state, str(user_message or "")) and self.llm is not None:
            # Deterministic parsing did not recognise this phrasing. This is
            # exactly the case the LLM earns its keep on: a paraphrased or
            # free-form customer. On the literal protocol it never fires.
            self._observe_with_llm(state, str(user_message or ""))
        # `maxItems` in the contract is 100, whatever the harness asks for.
        limit = min(top_k, MAX_RECOMMENDATIONS) if isinstance(top_k, int) and top_k > 0 else 10
        width = self._shortlist_size(state, turn, limit)
        ranked = self._rank(state, max(limit, LLM_RERANK_DEPTH if self._llm_rerank else 0))
        if self._llm_rerank and self.llm is not None and state.constraints:
            ranked = self._rerank_with_llm(state, ranked)
        ranked = ranked[:width]
        attribute, message = self._next_question(state, ranked, turn)
        if attribute is not None and attribute not in ATTRIBUTES:
            attribute = "other"
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            "usage": {
                "prompt_tokens": (self.llm.usage_snapshot()[0] - before[0]) if self.llm else 0,
                "completion_tokens": (self.llm.usage_snapshot()[1] - before[1]) if self.llm else 0,
            },
        }

    # -- dialogue understanding --------------------------------------------
    def _observe(self, state: SessionState, message: str) -> bool:
        """Parse one customer turn. Returns True if it matched a known form."""
        text = message.strip()
        if not text:
            return True

        nothing_more = RE_NOTHING_MORE.search(text)
        if nothing_more:
            # The customer has disclosed everything they can: further
            # questioning cannot sharpen the ranking any more.
            state.asked.add(nothing_more.group(1).lower())
            if nothing_more.group(1).lower() == "other":
                state.open_question_exhausted = True
            state.info_exhausted = True
            return True

        no_opinion = RE_NO_OPINION.search(text)
        if no_opinion:
            # The customer simply has no view on this attribute (a Boundary
            # turn). That is not the same as having nothing left to tell us,
            # so keep probing - just not about this attribute again.
            state.asked.add(no_opinion.group(1).lower())
            return True

        override = RE_OVERRIDE.search(text)
        if override:
            # The customer replaced an earlier preference: keep previously stated
            # requirements only as a weak prior and trust the new one instead.
            for constraint in state.constraints:
                constraint.demote(RETRACTED_WEIGHT)
            self._add_constraint(state, override.group(1), promote_existing=True)
            return True

        reveal = RE_REVEAL.search(text)
        if reveal:
            for piece in reveal.group(1).split("; "):
                self._add_constraint(state, piece)
            return True

        buying = RE_OPEN_BUYING.match(text)
        if buying:
            self._set_category(state, buying.group(1))
            self._add_constraint(state, buying.group(2))
            return True

        browsing = RE_OPEN_BROWSING.match(text)
        if browsing:
            self._set_category(state, browsing.group(1))
            return True

        generic = RE_OPEN_GENERIC.match(text)
        if generic:
            self._set_category(state, generic.group(1))
            self._add_constraint(state, generic.group(2))
            return True

        bare = RE_OPEN_BARE.match(text)
        if bare:
            self._set_category(state, bare.group(1))
            return True

        # Unrecognised phrasing (e.g. a paraphrased customer). Degrade
        # gracefully: recover the product type if we do not have one yet, keep
        # each clause as a weaker requirement, and bank the content words.
        if not state.category:
            self._infer_category(state, text)
        for clause in re.split(r"[.;!?]", text):
            clause = _normalize(clause).strip(" -,")
            if len(clause.split()) >= 3:
                state.add("phrase", clause, weight=FREE_TEXT_WEIGHT)
        state.free_tokens.update(_content_tokens(text))
        return False

    def _observe_with_llm(self, state: SessionState, message: str) -> None:
        """Recover structure from an unrecognised utterance. Never raises."""
        try:
            parsed = self.llm.complete_json(EXTRACT_SYSTEM, f"Customer message:\n{message}")
        except Exception:
            return
        if not isinstance(parsed, dict):
            return
        retracts = bool(parsed.get("retracts_earlier"))
        if retracts:
            for constraint in state.constraints:
                constraint.demote(RETRACTED_WEIGHT)
        product_type = parsed.get("product_type")
        if isinstance(product_type, str) and product_type.strip() and not state.category:
            # Map the model's wording onto a real catalog product type.
            self._infer_category(state, product_type)
        requirements = parsed.get("requirements")
        if isinstance(requirements, list):
            for requirement in requirements[:8]:
                if isinstance(requirement, str) and requirement.strip():
                    # Mirrors the deterministic override branch: a restated
                    # requirement must come back at full strength.
                    self._add_constraint(state, requirement, LLM_WEIGHT,
                                         promote_existing=retracts)
        if parsed.get("nothing_more_to_add"):
            state.info_exhausted = True

    def _rerank_with_llm(self, state: SessionState, ranked: list[str]) -> list[str]:
        """Semantic reorder of the head of the ranking. Falls back on any error."""
        head = ranked[:LLM_RERANK_DEPTH]
        if len(head) < 2:
            return ranked
        position = {asin: index for index, asin in enumerate(self.asins)}
        wants = "; ".join(constraint.text for constraint in state.constraints[:8])
        listing = "\n".join(
            f"{number}. {self.titles[position[asin]][:110]}"
            for number, asin in enumerate(head, start=1)
        )
        try:
            parsed = self.llm.complete_json(
                RERANK_SYSTEM,
                f"Customer wants: {wants}\nProduct type: {state.category or 'unspecified'}\n\n{listing}",
                max_tokens=200,
            )
        except Exception:
            return ranked
        if not isinstance(parsed, dict) or not isinstance(parsed.get("order"), list):
            return ranked
        reordered: list[str] = []
        seen: set[str] = set()
        for number in parsed["order"]:
            try:
                asin = head[int(number) - 1]
            except (ValueError, TypeError, IndexError):
                continue
            if asin not in seen:
                seen.add(asin)
                reordered.append(asin)
        # Anything the model dropped keeps its deterministic order behind.
        reordered.extend(asin for asin in ranked if asin not in seen)
        return reordered

    def _infer_category(self, state: SessionState, text: str) -> None:
        """Recover the product type from free prose by matching bucket names."""
        tokens = _content_tokens(text)
        if not tokens:
            return
        best_name: str | None = None
        best_key = (0.0, 0)
        for name, name_tokens in self._bucket_tokens.items():
            if not name_tokens:
                continue
            matched = len(name_tokens & tokens)
            if not matched:
                continue
            # Prefer well-covered names, then the most specific one: both
            # "shirts t-shirts" and "shirts tanks tops" are fully covered by
            # the latter's wording, and the longer match is the real type.
            key = (matched / len(name_tokens), matched)
            if key > best_key:
                best_name, best_key = name, key
        if best_name and best_key[0] >= FUZZY_CATEGORY_FLOOR:
            self._set_category(state, best_name)

    def _set_category(self, state: SessionState, category: str) -> None:
        category = _normalize(category)
        if not category:
            return
        if category != state.category:
            state.bucket = None
        state.category = category
        state.category_tokens = _content_tokens(category)

    def _add_constraint(
        self,
        state: SessionState,
        raw: str,
        weight: float = 1.0,
        promote_existing: bool = False,
    ) -> None:
        text = _normalize(raw).strip(" -;,.")
        if not text:
            return

        color = RE_COLOR_CONSTRAINT.match(text)
        if color:
            state.add("color", color.group(1).strip(), promote_existing=promote_existing)
            return

        budget = RE_BUDGET_CONSTRAINT.search(text)
        if budget:
            state.add(
                "budget",
                text,
                value=float(budget.group(1)),
                promote_existing=promote_existing,
            )
            return

        if text in MATERIALS:
            state.add("material", text, promote_existing=promote_existing)
            return

        state.add("phrase", text, weight, promote_existing=promote_existing)
        # A quoted requirement may also pin down a price point.
        price = RE_PRICE_IN_TEXT.search(text)
        if price and "budget" in text:
            state.add(
                "budget",
                text,
                value=float(price.group(1)),
                promote_existing=promote_existing,
            )

    # -- retrieval ----------------------------------------------------------
    def _candidates(self, state: SessionState) -> tuple[list[int], bool]:
        """Candidate pool: the stated product type, else the whole catalog."""
        if state.category:
            if state.bucket is None:
                state.bucket = self._resolve_bucket(state)
            if state.bucket:
                return state.bucket, True
        return range(len(self.asins)), False  # type: ignore[return-value]

    def _resolve_bucket(self, state: SessionState) -> list[int]:
        """Exact product-type match, else the closest-named types by token overlap."""
        bucket = self.buckets.get(state.category or "")
        if bucket:
            return bucket
        tokens = state.category_tokens
        if not tokens:
            return []
        scored: list[tuple[float, str]] = []
        for name, name_tokens in self._bucket_tokens.items():
            if not name_tokens:
                continue
            overlap = len(name_tokens & tokens) / len(name_tokens | tokens)
            if overlap >= 0.34:
                scored.append((overlap, name))
        if not scored:
            return []
        scored.sort(reverse=True)
        pool: list[int] = []
        for _, name in scored[:5]:
            pool.extend(self.buckets[name])
        return pool

    def _constraint_hits(self, index: int, state: SessionState) -> tuple[list[tuple[int, float]], float]:
        """Evidence this product satisfies each stated requirement.

        Returns one (constraint, strength) pair per satisfied requirement plus a
        small soft-preference bonus. Strengths are raw; `_rank` reweights them by
        how discriminative each requirement is across the candidate pool.
        """
        blob = self.blobs[index]
        hits: list[tuple[int, float]] = []
        tokens: set[str] | None = None
        for position, constraint in enumerate(state.constraints):
            if constraint.kind == "budget":
                strength = self._budget_score(index, constraint)
                if strength:
                    hits.append((position, constraint.weight * strength))
                continue
            if constraint.kind in ("material", "color"):
                if constraint.text in blob:
                    hits.append((position, constraint.weight * W_ATTRIBUTE_HIT))
                continue
            # Free-text requirement: exact quotation is the strongest evidence.
            if constraint.text and constraint.text in blob:
                length = min(len(constraint.tokens), W_PHRASE_TOKEN_CAP)
                hits.append((position, constraint.weight * (W_PHRASE_BASE + W_PHRASE_PER_TOKEN * length)))
                continue
            if not constraint.tokens:
                continue
            if tokens is None:
                tokens = self._blob_tokens(index)
            overlap = len(constraint.tokens & tokens) / len(constraint.tokens)
            if overlap >= 0.6:
                hits.append((position, constraint.weight * W_OVERLAP_STRONG * overlap * overlap))
            elif overlap:
                hits.append((position, constraint.weight * W_OVERLAP_WEAK * overlap))
        bonus = 0.0
        if state.free_tokens:
            if tokens is None:
                tokens = self._blob_tokens(index)
            bonus = W_SOFT_TOKEN * len(state.free_tokens & tokens)
        return hits, bonus

    def _budget_score(self, index: int, constraint: Constraint) -> float:
        price = self.prices[index]
        if price is None or not constraint.value:
            return 0.0
        ratio = abs(price - constraint.value) / constraint.value
        if ratio <= 0.08:
            return W_BUDGET_CLOSE
        if ratio <= 0.30:
            return W_BUDGET_NEAR
        return W_BUDGET_OFF

    def _rank(self, state: SessionState, limit: int) -> list[str]:
        candidates, bucketed = self._candidates(state)
        rows: list[tuple[int, float, list[tuple[int, float]]]] = []
        document_frequency = [0] * len(state.constraints)
        for index in candidates:
            base = self.priors[index]
            if not bucketed and state.category_tokens:
                base += self._category_score(index, state)
            hits, bonus = self._constraint_hits(index, state)
            for position, _ in hits:
                document_frequency[position] += 1
            rows.append((index, base + bonus, hits))

        # A requirement satisfied by most of the pool ("imported", "machine wash")
        # carries almost no signal; a rarely satisfied one is decisive.
        total = max(len(rows), 1)
        ceiling = max(math.log(total), 1.0)
        weights = [
            IDF_FLOOR + IDF_SPAN * min(1.0, math.log((total + 1.0) / (count + 0.5)) / ceiling)
            if count else 0.0
            for count in document_frequency
        ]
        scored: list[tuple[float, int]] = []
        for index, base, hits in rows:
            score = base
            for position, strength in hits:
                score += strength * weights[position]
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.asins[index] for _, index in scored[:limit]]

    def _category_score(self, index: int, state: SessionState) -> float:
        if self.coarse[index].lower() == state.category:
            return W_CATEGORY_EXACT
        tokens = self._blob_tokens(index)
        overlap = len(state.category_tokens & tokens) / len(state.category_tokens)
        return W_CATEGORY_OVERLAP * overlap

    def _shortlist_size(self, state: SessionState, turn: int, limit: int) -> int:
        """How many products to put in front of the customer this turn.

        While the customer still has undisclosed requirements, the next answer
        is worth more than a wider guess: we surface only the single best match
        and spend the turn on a question. Once questioning can no longer sharpen
        the ranking - the customer says they have nothing more to add, or the
        turn budget is running down - we present the full shortlist.
        """
        if state.info_exhausted or turn >= 5 or self._evidence_saturated(state):
            return limit
        if turn >= 3:
            return min(3, limit)
        return 1

    @staticmethod
    def _evidence_saturated(state: SessionState) -> bool:
        """Whether the dialogue already contains enough independent evidence.

        A known product type plus four unique requirements is sufficiently
        specific to expose the complete shortlist without waiting for a later
        turn. ``SessionState.add`` deduplicates requirements by normalized kind
        and text, so repeated wording cannot satisfy this gate.
        """
        return bool(
            state.category
            and len(state.constraints) >= EVIDENCE_SATURATION_MIN_CONSTRAINTS
        )

    # -- clarification policy ----------------------------------------------
    def _next_question(self, state: SessionState, ranked: list[str], turn: int) -> tuple[str | None, str]:
        """Ask the question with the highest expected information gain.

        An open "anything else that matters" probe dominates while the customer
        still has undisclosed requirements, because it is not restricted to a
        single attribute. Once that is spent we fall back to probing specific
        attributes we have not covered yet.
        """
        if not ranked:
            return "category", "I couldn't find a match yet — what type of item are you after?"
        if len(ranked) == 1:
            preface = "Here's my closest match so far. "
        else:
            preface = f"Here are my top {len(ranked)} matches. "
        if not state.open_question_exhausted:
            state.asked.add("other")
            return "other", preface + QUESTION_TEXT["other"]
        for attribute in PROBE_ORDER:
            if attribute not in state.asked:
                state.asked.add(attribute)
                return attribute, preface + QUESTION_TEXT[attribute]
        return None, preface + "Let me know if any of these look right."
