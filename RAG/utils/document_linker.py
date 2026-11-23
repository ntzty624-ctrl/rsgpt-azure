import os
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DocumentLinker:

    def __init__(self, config):
        self.config = config

    def get_sharepoint_link(self, document_name: str, doc_metadata: dict) -> Optional[str]:
        """
        Args:
            document_name: Name of the document
            doc_metadata: Dict keyed by file_name containing chunk metadata (sharepoint_item_url)
        Returns:
            str: Full SharePoint URL or None if not found
        """
        meta = doc_metadata.get(document_name, {})
        return meta.get("sharepoint_item_url") or None

    def add_links_to_response(self, response_text: str, doc_metadata: dict = None) -> str:
        """
        Args:
            response_text: Response text with document references
        Returns:
            str: Response text with added hyperlinks
        """
        if doc_metadata is None:
            doc_metadata = {}

        # First pass: Link quoted PDF filenames
        response_text = self._link_quoted_pdfs(response_text, doc_metadata)

        # Second pass: Process line by line for structured content
        lines = response_text.splitlines()
        updated_lines = self._process_response_lines(lines, doc_metadata)

        return "\n".join(updated_lines)

    def _link_quoted_pdfs(self, text: str, doc_metadata: dict) -> str:
        """Replace quoted PDF names with hyperlinks."""
        def replace_pdf(match):
            doc_name = match.group(1)
            link = self.get_sharepoint_link(doc_name, doc_metadata)
            if link:
                return f'<a href="{link}" target="_blank">{doc_name}</a>'
            return match.group(0)

        return re.sub(r'"([^"]+\.pdf)"', replace_pdf, text)

    def _process_response_lines(self, lines: list[str], doc_metadata: dict) -> list[str]:

        updated_lines = []
        current_doc = None

        # Regex patterns
        doc_header_pattern = re.compile(
            r'\*\*From\s+<a [^>]+>([^<]+\.pdf)</a>\*\*:',
            re.IGNORECASE
        )
        suggested_doc_pattern = re.compile(
            r'^- <a href="([^"]+)">([^<]+)</a>, pages ([\d,\s]+)',
            re.IGNORECASE
        )
        page_pattern = re.compile(
            r'(?:\*\*)?Page (\d+)(?:\*\*)?(?=\s*[:\-─€"])',
            re.IGNORECASE
        )

        for line in lines:
            # Check for document header
            doc_match = doc_header_pattern.search(line)
            if doc_match:
                current_doc = doc_match.group(1)
                updated_lines.append(line)
                continue

            # Handle suggested documents with page numbers
            sugg_match = suggested_doc_pattern.search(line)
            if sugg_match:
                line = self._process_suggested_doc_line(sugg_match)
                updated_lines.append(line)
                continue

            # Add page links in details section
            if current_doc:
                line = self._add_page_links(line, current_doc, page_pattern, doc_metadata)

            updated_lines.append(line)

        return updated_lines

    def _process_suggested_doc_line(self, match) -> str:
        """Process a suggested document line to add page links."""
        doc_link = match.group(1)
        doc_name = match.group(2)
        pages = [p.strip() for p in match.group(3).split(',')]

        linked_pages = [
            f'<a href="{doc_link}?page={p}" target="_blank">{p}</a>'
            for p in pages
        ]

        return (f'- <a href="{doc_link}" target="_blank">{doc_name}</a>, '
                f'pages {", ".join(linked_pages)}')

    def _add_page_links(self, line: str, doc_name: str, page_pattern: re.Pattern,
                        doc_metadata: dict) -> str:

        def replace_page(match):
            page_num = match.group(1)
            link = self.get_sharepoint_link(doc_name, doc_metadata)
            if link:
                return f'<a href="{link}#page={page_num}" target="_blank">Page {page_num}</a>'
            return f'Page {page_num}'

        return page_pattern.sub(replace_page, line)
