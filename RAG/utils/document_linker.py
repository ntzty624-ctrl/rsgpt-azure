import html
import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# [D1] or [D1, D2]
_TAG_RE = re.compile(r"\[(D\d+(?:\s*,\s*D\d+)*)\]")
# "page 3", "Pages 3, 5 and 7", "**Page 4**"
_PAGE_RE = re.compile(r"\b([Pp]ages?)(\*\*)?\s+(\d+(?:\s*(?:,|and|&|-|–)\s*\d+)*)")


class DocumentLinker:
    """Turns the model's [Dn] tags into document names and SharePoint links.

    Names and links always come from chunk metadata, never from the model, so they
    are exact. Tags the model invents (not in the context) are removed and counted.
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def link(meta: dict, label: str = None, page: str = None) -> str:
        name = html.escape(label or meta.get("source") or meta.get("file_name") or "Unknown")
        url = meta.get("sharepoint_item_url")
        if not url:
            return name
        href = html.escape(url + (f"#page={page}" if page else ""), quote=True)
        return f'<a href="{href}" target="_blank">{name}</a>'

    def render(self, response_text: str, tag_map: Dict[str, dict]) -> Tuple[str, dict]:
        """Replace tags with linked names and link page numbers.

        Returns (text, stats) where stats = {"cited_docs", "citations", "invalid_citations"}.
        """
        cited: List[str] = []
        stats = {"citations": 0, "invalid_citations": 0}
        current = None  # document the current line / section refers to
        out_lines = []

        for line in response_text.splitlines():
            tags = [t.strip() for m in _TAG_RE.finditer(line) for t in m.group(1).split(",")]
            valid = [t for t in tags if t in tag_map]
            if valid:
                current = valid[-1]
            elif tags:
                # Only invented tags: drop the reference line / header instead of leaving it nameless
                stats["citations"] += len(tags)
                stats["invalid_citations"] += len(tags)
                current = None
                if line.lstrip().startswith(("-", "*")):
                    continue
            elif line.startswith("#"):
                current = None

            if current and _PAGE_RE.search(line):
                meta = tag_map[current]
                line = _PAGE_RE.sub(lambda m: m.group(1) + (m.group(2) or "") + " " + re.sub(
                    r"\d+", lambda p: self.link(meta, p.group(0), p.group(0)), m.group(3)), line)

            def replace_tag(m):
                parts = []
                for tag in (t.strip() for t in m.group(1).split(",")):
                    stats["citations"] += 1
                    if tag in tag_map:
                        if tag not in cited:
                            cited.append(tag)
                        parts.append(self.link(tag_map[tag]))
                    else:
                        stats["invalid_citations"] += 1
                return ", ".join(parts)

            out_lines.append(_TAG_RE.sub(replace_tag, line))

        if stats["invalid_citations"]:
            logger.warning(f"Model cited {stats['invalid_citations']} tag(s) not in the context")
        stats["cited_docs"] = len(cited)
        stats["cited"] = cited
        return "\n".join(out_lines), stats

    def matches_section(self, match, cited_numbers=None) -> str:
        """'Document Matches' section built from the index (exact / not found + nearest / partial)."""
        if match is None or (match.match_type == "none" and not match.not_found):
            return ""
        cited_numbers = set(cited_numbers or [])

        def entry_link(e) -> str:
            return self.link({"source": e.file_name or e.document_number,
                              "sharepoint_item_url": e.sharepoint_item_url})

        lines = ["## 3. Document Matches:"]
        for e in match.exact:
            note = "" if e.document_number in cited_numbers else " (requested; not used in the answer above)"
            lines.append(f"- Requested: {entry_link(e)}{note}")
        if match.not_found:
            missing = ", ".join(f'"{html.escape(n)}"' for n in match.not_found)
            nearest = ", ".join(entry_link(e) for e in match.nearest) or "none"
            lines.append(f"- {missing} not found. Nearest: {nearest}")
        elif match.nearest:
            lines.append(f"- No exact match. Nearest: {', '.join(entry_link(e) for e in match.nearest)}")
        if match.partial:
            tokens = ", ".join(f'"{html.escape(t)}"' for t in dict.fromkeys(match.partial_tokens))
            shown = match.partial[:self.config.id_candidates_max]
            more = len(match.partial) - len(shown)
            lines.append(f"- Possible documents for {tokens}: " + ", ".join(entry_link(e) for e in shown)
                         + (f" (+{more} more)" if more > 0 else ""))
        return "\n".join(lines) if len(lines) > 1 else ""
