import os
import re
import time
import logging
import threading
from typing import Dict, List, Tuple

from .analytics import log_event

logger = logging.getLogger(__name__)

_ENTRY_RE = re.compile(r"^-\s*(.+?)\s*:\s*(.+?)\s*$")


class GlossaryManager:
    """Loads the hand-maintained glossary and reloads it when the file changes."""

    def __init__(self, config):
        self.config = config
        self.glossary_dict: Dict[str, str] = {}
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
            # tokens (document numbers such as C-ST-FO-4110) are never expanded.
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
                      load_ms=round((time.perf_counter() - t) * 1000), trigger=trigger)
            return True

    @staticmethod
    def _parse(content: str) -> Tuple[Dict[str, str], List[str]]:
        glossary: Dict[str, str] = {}
        unparsed: List[str] = []
        for line in content.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = _ENTRY_RE.match(line)
            if m:
                glossary[m.group(1)] = m.group(2)
            else:
                unparsed.append(line)
        return glossary, unparsed

    def expand_acronyms(self, text: str) -> Tuple[str, List[str]]:
        """Expand acronyms as 'ST (Standard)', keeping the acronym itself.

        Returns (expanded_text, matched_keys).
        """
        if not self._pattern:
            return text, []

        matched: List[str] = []

        def repl(m):
            key = m.group(1)
            if key not in matched:
                matched.append(key)
            return f"{key} ({self.glossary_dict[key]})"

        expanded = self._pattern.sub(repl, text)
        if matched:
            logger.debug(f"Expanded acronyms in query: {matched}")
        return expanded, matched

    def get_glossary_text(self, keys: List[str] = None) -> str:
        """Glossary lines for the prompt — only the given keys (none → empty)."""
        if not keys:
            return ""
        lines = [f"- {k}: {self.glossary_dict[k]}" for k in keys if k in self.glossary_dict]
        return "Glossary:\n" + "\n".join(lines) if lines else ""
