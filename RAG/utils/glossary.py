import os
import re
import time
import logging
import threading
from typing import Dict, List, Tuple

from .analytics import log_event
from .document_index import QUERY_NUMBER_RE

logger = logging.getLogger(__name__)

_ENTRY_RE = re.compile(r"^-\s*(.+?)\s*:\s*(.+?)\s*$")


class GlossaryManager:
    """Loads the hand-maintained glossary and reloads it when the file changes.

    Abbreviations are treated as *possible* meanings: the question is never rewritten.
    Meanings are offered to the model as hints, and (mode "hint") appended to the
    searched text.
    """

    def __init__(self, config):
        self.config = config
        self.glossary_dict: Dict[str, List[str]] = {}
        self._pattern = None
        self._mtime = None
        self._missing_logged = False
        self._lock = threading.Lock()
        self.reload_if_changed()

    def reload_if_changed(self) -> bool:
        """Reload the glossary if the file's modified time changed. Cheap (one stat call)."""
        path = self.config.get_glossary_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if not self._missing_logged:
                logger.error(f"Glossary file not found: {path}")
                log_event("glossary_load", entries=0, unparsed_lines=0, load_ms=0, trigger="missing")
                self._missing_logged = True
            return False

        if mtime == self._mtime:
            return False

        with self._lock:
            if mtime == self._mtime:
                return False
            t = time.perf_counter()
            trigger = "cold" if self._mtime is None else "file_changed"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading glossary file: {str(e)}")
                return False

            glossary, unparsed = self._parse(content)
            self.glossary_dict = glossary
            # Longest keys first so e.g. "STE" wins over "ST". Keys inside hyphenated
            # tokens (document numbers such as C-ST-FO-4110) never match.
            keys = sorted(glossary, key=len, reverse=True)
            self._pattern = re.compile(
                r'(?<![\w-])(' + '|'.join(re.escape(k) for k in keys) + r')(?![\w-])'
            ) if keys else None
            self._mtime = mtime
            self._missing_logged = False

            if unparsed:
                logger.warning(f"Glossary: {len(unparsed)} line(s) not in '- KEY: value' format, "
                               f"e.g. {unparsed[:3]}")
            log_event("glossary_load", entries=len(glossary), unparsed_lines=len(unparsed),
                      multi_meaning_keys=sum(len(v) > 1 for v in glossary.values()),
                      load_ms=round((time.perf_counter() - t) * 1000), trigger=trigger)
            return True

    @staticmethod
    def _parse(content: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """Repeated keys keep every meaning, e.g. ST: Standard and ST: Station."""
        glossary: Dict[str, List[str]] = {}
        unparsed: List[str] = []
        for line in content.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = _ENTRY_RE.match(line)
            if m:
                meanings = glossary.setdefault(m.group(1), [])
                if m.group(2) not in meanings:
                    meanings.append(m.group(2))
            else:
                unparsed.append(line)
        return glossary, unparsed

    def find_acronyms(self, text: str) -> List[str]:
        """Glossary keys in the text, skipping anything that looks like a document number
        (including spaced forms such as 'C ST FO 4110' or 'SIN A01 002')."""
        if not self._pattern:
            return []
        protected = [m.span() for m in QUERY_NUMBER_RE.finditer(text)]
        found: List[str] = []
        for m in self._pattern.finditer(text):
            if any(a <= m.start() < b for a, b in protected):
                continue
            if m.group(1) not in found:
                found.append(m.group(1))
        if found:
            logger.debug(f"Abbreviations in query: {found}")
        return found

    def search_text(self, text: str, keys: List[str], mode: str = "hint") -> str:
        """Text to embed. 'hint' appends possible meanings; 'off' leaves the text as typed."""
        if mode != "hint" or not keys:
            return text
        hints = "; ".join(f"{k} = {' / '.join(self.glossary_dict[k])}" for k in keys if k in self.glossary_dict)
        return f"{text} — possible meanings: {hints}" if hints else text

    def get_glossary_text(self, keys: List[str] = None) -> str:
        """Possible meanings for the prompt — only the given keys (none → empty)."""
        if not keys:
            return ""
        lines = [
            f"- {k}: possibly " + " or ".join(f'"{v}"' for v in self.glossary_dict[k])
            for k in keys if k in self.glossary_dict
        ]
        if not lines:
            return ""
        return ("Possible meanings of abbreviations in the question "
                "(they may not apply; decide from the context):\n" + "\n".join(lines))
