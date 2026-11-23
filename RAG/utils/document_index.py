"""
In-memory index of documents (one entry per document) used for document-number and
title lookups.

Where the entries come from is pluggable (the "source"). The base EmptySource has no
entries, which turns number matching and title search off (old behaviour).
"""
import re
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .analytics import log_event

logger = logging.getLogger(__name__)

# Same three formats as rs-gpt sync_compare._DOC_NUMBER_RE (SIN-A01-002, C-ST-FO-4110,
# S005), searched anywhere in a query and allowing spaces / no separators
# ("sin a01 002", "cstfo4110"). Hits are only trusted if they exist in the index.
_SEP = r"[-\s_]?"
_NUMBER_BODY = (
    r"(?P<sin>SIN" + _SEP + r"[A-Z]\d{2})" + _SEP + r"(?P<sin_n>\d{3})"
    r"|(?P<c>[A-Z]" + _SEP + r"[A-Z]{2}" + _SEP + r"[A-Z]{2})" + _SEP + r"(?P<c_n>\d+)"
    r"|(?P<s>S)(?P<s_n>\d{3,4})"
)
QUERY_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + _NUMBER_BODY + r")(?![A-Za-z0-9])", re.IGNORECASE)
_FULL_NUMBER_RE = re.compile(r"^(?:" + _NUMBER_BODY + r")$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_number(text: str) -> str:
    """'sin-a01 002' -> 'SINA01002' (case and separators ignored)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _prefix_number(match) -> Tuple[str, Optional[int]]:
    """From a number regex match: ('SINA01', 2) for SIN-A01-002, ('CSTFO', 4110), ('S', 5)."""
    for key in ("sin", "c", "s"):
        if match.group(key):
            return normalize_number(match.group(key)), int(match.group(key + "_n"))
    return "", None


@dataclass
class IndexEntry:
    document_number: str
    file_name: str
    title: str = ""
    sharepoint_item_url: str = ""
    norm: str = ""
    prefix: str = ""
    number: Optional[int] = None
    segments: frozenset = frozenset()

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        num = d.get("document_number", "")
        norm = normalize_number(num)
        m = _FULL_NUMBER_RE.match(num.strip())
        prefix, number = _prefix_number(m) if m else (norm, None)
        segments = {s for s in re.split(r"[-\s_]+", num.upper()) if s}
        trailing = re.search(r"(\d+)$", num)
        if trailing:
            segments.add(trailing.group(1))  # S005 -> also "005"
        return cls(
            document_number=num,
            file_name=d.get("file_name", ""),
            title=d.get("title", ""),
            sharepoint_item_url=d.get("sharepoint_item_url", ""),
            norm=norm, prefix=prefix, number=number, segments=frozenset(segments),
        )


@dataclass
class MatchResult:
    """Document numbers found in a query, checked against the real index."""
    exact: List[IndexEntry] = field(default_factory=list)
    partial: List[IndexEntry] = field(default_factory=list)
    nearest: List[IndexEntry] = field(default_factory=list)
    not_found: List[str] = field(default_factory=list)   # typed numbers that are not in the index
    partial_tokens: List[str] = field(default_factory=list)

    @property
    def match_type(self) -> str:
        if self.exact:
            return "exact"
        if self.partial:
            return "partial"
        if self.nearest:
            return "nearest"
        return "none"

    def to_event(self) -> dict:
        return {
            "match_type": self.match_type,
            "exact": [e.document_number for e in self.exact],
            "partial": [e.document_number for e in self.partial],
            "nearest": [e.document_number for e in self.nearest],
            "not_found": self.not_found,
        }


class EmptySource:
    """No index entries — number matching and title search are off."""
    name = "none"

    def load(self, previous_marker: Optional[str]):
        """Return (entries | None when unchanged, marker, extra stats for the index_load event)."""
        return [], None, {}

    def title_search(self, index: "DocumentIndex", query_embedding: List[float], k: int) -> List[Tuple[str, float]]:
        """Return [(document_number, similarity 0..1)], best first."""
        return []


class DocumentIndex:

    def __init__(self, config, source=None):
        self.config = config
        self.source = source or EmptySource()
        self.entries: List[IndexEntry] = []
        self.by_norm: Dict[str, IndexEntry] = {}
        self.by_number: Dict[str, IndexEntry] = {}
        self.marker: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self._last_check: Optional[float] = None
        self._lock = threading.Lock()

    # ── Loading ──────────────────────────────────────────────────────────

    @property
    def age_s(self) -> Optional[int]:
        return None if self.loaded_at is None else round(time.time() - self.loaded_at)

    def ensure_fresh(self) -> None:
        """Load on first use, then re-check every index_refresh_s seconds."""
        now = time.time()
        if self._last_check is not None and now - self._last_check < self.config.index_refresh_s:
            return
        with self._lock:
            if self._last_check is not None and now - self._last_check < self.config.index_refresh_s:
                return
            self._reload("cold" if self._last_check is None else "refresh")
            self._last_check = time.time()

    def _reload(self, trigger: str) -> None:
        t = time.perf_counter()
        try:
            entries, marker, extra = self.source.load(self.marker if self.entries else None)
        except Exception as e:
            logger.warning(f"Document index load failed ({self.source.name}): {e}", exc_info=True)
            log_event("index_load", source=self.source.name, trigger=trigger, error=type(e).__name__,
                      entries=len(self.entries), load_ms=round((time.perf_counter() - t) * 1000))
            return
        if entries is None:
            trigger = "unchanged"
        else:
            self._set_entries(entries)
            self.marker = marker
            self.loaded_at = time.time()
        log_event("index_load", source=self.source.name, trigger=trigger, entries=len(self.entries),
                  load_ms=round((time.perf_counter() - t) * 1000), **extra)

    def _set_entries(self, entries: List[dict]) -> None:
        parsed = [IndexEntry.from_dict(d) for d in entries if d.get("document_number")]
        self.entries = parsed
        self.by_norm = {e.norm: e for e in parsed}
        self.by_number = {e.document_number: e for e in parsed}

    # ── Number matching ──────────────────────────────────────────────────

    def match_numbers(self, query: str) -> MatchResult:
        """Exact → partial → nearest matching of document numbers in the query.

        - exact:   a full number (any case / separators) that exists
        - partial: a token with a digit equal to one part of real numbers ("001", "A01", "4110")
        - nearest: a full number that does not exist → same prefix, closest number;
                   or a bare zero-padded number ("001") matching no part → closest numbers
        """
        result = MatchResult()
        if not self.entries:
            return result

        full_hits = []
        prefixes = {e.prefix for e in self.entries}
        for m in QUERY_NUMBER_RE.finditer(query):
            norm = normalize_number(m.group(0))
            entry = self.by_norm.get(norm)
            if entry:
                full_hits.append(m)
                if entry not in result.exact:
                    result.exact.append(entry)
                continue
            prefix, number = _prefix_number(m)
            if prefix not in prefixes:
                continue  # looks like a number ("about 5") but no such series exists
            full_hits.append(m)
            result.not_found.append(m.group(0))
            result.nearest.extend(e for e in self._closest(number, prefix) if e not in result.nearest)

        # Partial: tokens outside the full-number hits that contain a digit
        spans = [h.span() for h in full_hits]
        scores: Dict[str, int] = {}
        bare_misses: List[str] = []
        for tm in _TOKEN_RE.finditer(query):
            if any(a <= tm.start() < b for a, b in spans):
                continue
            token = tm.group(0).upper()
            if not any(c.isdigit() for c in token):
                continue
            hits = [e for e in self.entries if token in e.segments]
            if hits:
                result.partial_tokens.append(token)
                for e in hits:
                    scores[e.document_number] = scores.get(e.document_number, 0) + 1
            elif token.isdigit() and token.startswith("0") and 3 <= len(token) <= 4:
                bare_misses.append(token)

        exact_nums = {e.document_number for e in result.exact}
        ranked = sorted((n for n in scores if n not in exact_nums), key=lambda n: (-scores[n], n))
        result.partial = [self.by_number[n] for n in ranked]

        if not result.exact and not result.partial:
            for token in bare_misses:
                result.nearest.extend(
                    e for e in self._closest(int(token), None, self.config.nearest_max_distance)
                    if e not in result.nearest
                )
        result.nearest = result.nearest[:self.config.id_candidates_max]
        return result

    def _closest(self, number: Optional[int], prefix: Optional[str], max_distance: int = None) -> List[IndexEntry]:
        """Entries with the numerically closest number (all ties), optionally same prefix only."""
        if number is None:
            return []
        pool = [e for e in self.entries if e.number is not None and (prefix is None or e.prefix == prefix)]
        if not pool:
            return []
        best = min(abs(e.number - number) for e in pool)
        if max_distance is not None and best > max_distance:
            return []
        return sorted((e for e in pool if abs(e.number - number) == best), key=lambda e: e.document_number)

    # ── Title search ─────────────────────────────────────────────────────

    def search_titles(self, query_embedding: List[float], k: int = None) -> List[Tuple[IndexEntry, float]]:
        if not self.entries:
            return []
        hits = self.source.title_search(self, query_embedding, k or self.config.title_search_k)
        return [(self.by_number[n], score) for n, score in hits if n in self.by_number]


class CollectionSource:
    """Index entries from the document_index collection written by rs-gpt (document_index.py).

    Only metadatas are downloaded (~600 small rows); embeddings stay in Chroma and title
    search is a query on that collection. Reloads only when the collection's
    updated_at marker changes.
    """
    name = "collection"

    def __init__(self, chromadb_client, config):
        self.client = chromadb_client.client
        self.collection_name = config.chromadb_index_collection
        self._collection = None

    def load(self, previous_marker: Optional[str]):
        t = time.perf_counter()
        col = self.client.get_collection(name=self.collection_name)  # re-fetched to see a new updated_at
        self._collection = col
        marker = (col.metadata or {}).get("updated_at")
        check_ms = round((time.perf_counter() - t) * 1000)
        if previous_marker is not None and marker == previous_marker:
            return None, marker, {"check_ms": check_ms, "index_updated_at": marker}
        t = time.perf_counter()
        rows = col.get(include=["metadatas"])["metadatas"]
        return rows, marker, {"check_ms": check_ms, "read_ms": round((time.perf_counter() - t) * 1000),
                              "index_updated_at": marker}

    def title_search(self, index: DocumentIndex, query_embedding: List[float], k: int) -> List[Tuple[str, float]]:
        col = self._collection
        if col is None:
            return []
        r = col.query(query_embeddings=[query_embedding], n_results=k, include=["metadatas", "distances"])
        space = (col.metadata or {}).get("hnsw:space", "l2")
        hits = []
        for meta, dist in zip(r["metadatas"][0], r["distances"][0]):
            # Chroma l2 is squared distance; for unit-length embeddings cos = 1 - d/2
            score = 1 - dist if space == "cosine" else 1 - dist / 2
            hits.append((meta.get("document_number"), max(0.0, score)))
        return hits


def create_document_index(config, chromadb_client=None, openai_client=None) -> DocumentIndex:
    """Build the DocumentIndex with the configured source (this branch: document_index collection)."""
    if chromadb_client is None:
        return DocumentIndex(config, EmptySource())
    return DocumentIndex(config, CollectionSource(chromadb_client, config))
