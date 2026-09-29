"""Title handling: franchise grouping and fuzzy title -> anime ID resolution.

MAL's canonical titles are romaji ("Shingeki no Kyojin"). English titles and synonyms come
from the optional Jikan enrichment; when present they resolve too ("Attack on Titan", "DN").
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass

# Trailing sequel markers: "Season 3", "3rd Season", "Part 2", "II", "Movie ...", "2".
_SEQUEL_SUFFIX = re.compile(
    r"\s+(\d+(st|nd|rd|th)\s+season|(first|second|third|fourth|fifth|final)\s+season"
    r"|season\s*\d*|part\s*\d+|movie.*"
    r"|specials?|ova|ona|recap|ii+|iv|v|\d+)$",
    re.IGNORECASE,
)
_TRAILING_SYMBOLS = re.compile(r"[^\w\s']+$")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize(text: str) -> str:
    """Casefold, strip accents and symbols: 'Yuru Camp△' -> 'yuru camp'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


def franchise_key(title: str) -> str:
    """Group seasons/movies/spin-offs of one franchise under a single key.

    'Shingeki no Kyojin Season 3 Part 2' -> 'shingeki no kyojin'
    'Hajime no Ippo: New Challenger'     -> 'hajime no ippo'
    'Kingdom 6th Season'                 -> 'kingdom'
    A heuristic: good enough to stop results being five seasons of the same show.
    """
    base = title.split(":")[0].strip()
    previous = None
    while previous != base:  # peel suffixes and symbols: 'Yuru Camp△ Season 2 Specials'
        previous = base
        base = _TRAILING_SYMBOLS.sub("", _SEQUEL_SUFFIX.sub("", base)).strip()
    return normalize(base) or normalize(title)


@dataclass(frozen=True)
class TitleEntry:
    anime_id: int
    title: str
    members: int
    aliases: tuple[str, ...] = ()  # English title, synonyms
    english: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.title, *self.aliases)


class TitleIndex:
    """In-memory title lookup (10k titles: fast enough, no extra service)."""

    def __init__(self, entries: list[TitleEntry]) -> None:
        # Popular entries first, so ties resolve to the title people most likely mean.
        self._entries = sorted(entries, key=lambda e: -e.members)
        self._by_norm: dict[str, TitleEntry] = {}
        for entry in self._entries:
            for name in entry.names:
                self._by_norm.setdefault(normalize(name), entry)
        self._norms = list(self._by_norm)
        self._normalized = [(normalize(n), e) for e in self._entries for n in e.names]

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, query: str, cutoff: float = 0.75) -> TitleEntry | None:
        """Exact (normalized) match, then prefix match, then fuzzy match."""
        q = normalize(query)
        if not q:
            return None
        if q in self._by_norm:
            return self._by_norm[q]
        for norm, entry in self._by_norm.items():  # popularity order
            if norm.startswith(q):
                return entry
        close = difflib.get_close_matches(q, self._norms, n=1, cutoff=cutoff)
        return self._by_norm[close[0]] if close else None

    def suggest(self, query: str, limit: int = 10) -> list[TitleEntry]:
        """Autocomplete: substring matches in popularity order."""
        q = normalize(query)
        if not q:
            return self._entries[:limit]
        out: dict[int, TitleEntry] = {}  # one row per anime even if several names match
        for norm, entry in self._normalized:
            if q in norm:
                out.setdefault(entry.anime_id, entry)
                if len(out) == limit:
                    break
        return list(out.values())


def same_franchise(a: str, b: str) -> bool:
    """Keys match, or one is a word-prefix of the other: 'non non biyori' ~
    'non non biyori repeat'. Short keys (one word) must match exactly to avoid
    'monster' swallowing 'monster musume'-style collisions of unrelated shows."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return " " in shorter and longer.startswith(shorter + " ")
