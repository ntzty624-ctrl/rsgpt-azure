import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from ..config import Config
from ..utils.glossary import GlossaryManager
from ..utils.document_linker import DocumentLinker
from ..utils.document_index import create_document_index, MatchResult
from ..clients.openai_client import OpenAIClient
from ..clients.chromadb_client import ChromaDBClient
from ..prompts import PROMPT_TEMPLATE
from ..utils.analytics import log_event

logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class RAGService:
    """Main service for RAG (Retrieval-Augmented Generation) operations."""

    def __init__(self, config: Config):
        """
        Args:
            config: Application configuration
        """
        self.config = config
        self.glossary_manager = GlossaryManager(config)
        self.document_linker = DocumentLinker(config)
        self.openai_client = OpenAIClient(config)
        self.chromadb_client = ChromaDBClient(config)
        self.document_index = create_document_index(config, self.chromadb_client, self.openai_client)

    def process_query(self, query_text: str) -> str:
        """
        Process a user query and return formatted response with document links.
        Args:
            query_text: User's query
        Returns:
            str: Formatted response with document suggestions and links
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        # One analytics event per request: what happened, how many, how long.
        event = {"request_id": uuid.uuid4().hex, "query_chars": len(query_text), "route": "rag"}
        timings: Dict[str, int] = {}
        event["timings_ms"] = timings
        t_total = time.perf_counter()
        stage = "retrieve"

        try:
            retrieval = self.retrieve(query_text, event, timings)
            docs = retrieval["docs"]

            if not docs:
                event["route"] = "no_results"
                return "No relevant documents found."

            # Generate response
            stage = "llm"
            t = time.perf_counter()
            response_text, tokens = self._generate_response(
                docs, retrieval["expanded_query"], retrieval["glossary_keys"]
            )
            timings["llm"] = _ms(t)
            event.update(tokens)
            stage = "links"

            # Build metadata lookup for document linker
            doc_metadata = {}
            for doc in docs:
                source = doc["metadata"].get("source", "")
                if source and source not in doc_metadata:
                    doc_metadata[source] = doc["metadata"]

            # Add SharePoint links
            final_response = self.document_linker.add_links_to_response(response_text, doc_metadata)

            return final_response

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            event["route"] = "error"
            log_event("rag_error", request_id=event["request_id"], stage=stage, error=type(e).__name__)
            raise
        finally:
            timings["total"] = _ms(t_total)
            log_event("rag_query", **event)

    def retrieve(self, query_text: str, event: dict = None, timings: Dict[str, int] = None) -> Dict[str, Any]:
        """Glossary → number matching → embedding → searches. No LLM call.

        Returns {"docs", "match", "expanded_query", "glossary_keys"} and fills event / timings.
        """
        event = event if event is not None else {}
        timings = timings if timings is not None else {}

        # Document numbers are matched on the raw query, before glossary expansion
        t = time.perf_counter()
        self.document_index.ensure_fresh()
        match = self.document_index.match_numbers(query_text)
        timings["number_match"] = _ms(t)
        event.update(match.to_event())
        event["index_source"] = self.document_index.source.name
        event["index_entries"] = len(self.document_index.entries)
        event["index_age_s"] = self.document_index.age_s

        # Expand acronyms in query (reloads the glossary first if the file changed)
        t = time.perf_counter()
        self.glossary_manager.reload_if_changed()
        expanded_query, glossary_keys = self.glossary_manager.expand_acronyms(query_text)
        timings["glossary"] = _ms(t)
        event["glossary_matched"] = len(glossary_keys)
        logger.debug(f"Expanded query: {expanded_query}")

        # Generate embedding for query
        t = time.perf_counter()
        query_embedding = self.openai_client.generate_embedding(expanded_query)
        timings["embed"] = _ms(t)

        # Search for relevant documents
        t = time.perf_counter()
        docs = self._search_documents(query_embedding, timings, match, event)
        timings["search_total"] = _ms(t)
        event["chunks_retrieved"] = len(docs)
        event["docs_in_context"] = len({d["metadata"].get("source") for d in docs})

        return {"docs": docs, "match": match, "expanded_query": expanded_query, "glossary_keys": glossary_keys}

    def _timed_query(self, timings: Dict[str, int], key: str, *args, **kwargs) -> Dict[str, Any]:
        t = time.perf_counter()
        try:
            return self.chromadb_client.query_collection(*args, **kwargs)
        finally:
            timings[key] = _ms(t)

    def _targeted_doc_numbers(self, match: Optional[MatchResult]) -> List[str]:
        """Documents to pull chunks from first: exact matches, then top partial matches."""
        if not match:
            return []
        nums = [e.document_number for e in match.exact]
        nums += [e.document_number for e in match.partial[:self.config.id_candidates_max]]
        return list(dict.fromkeys(nums))

    def _search_targeted(self, query_embedding: List[float], timings: Dict[str, int],
                         match: Optional[MatchResult], event: dict) -> Dict[str, Any]:
        """Most relevant chunks *within* documents matched by number. Never fails the request."""
        doc_numbers = self._targeted_doc_numbers(match)
        event["targeted_docs"] = len(doc_numbers)
        if not doc_numbers:
            return {}
        try:
            return self._timed_query(
                timings, "doc_number_search", query_embedding,
                n_results=self.config.doc_number_n_results,
                where={"$and": [{"document_number": {"$in": doc_numbers}}, {"active": True}]},
            )
        except Exception as e:
            logger.warning(f"Document number search failed: {e}", exc_info=True)
            event["doc_number_search_failed"] = True
            return {}

    def _search_documents(self, query_embedding: List[float], timings: Dict[str, int] = None,
                          match: Optional[MatchResult] = None, event: dict = None) -> List[Dict[str, Any]]:
        """
        Args:
            query_embedding: Query embedding vector
            timings: Dict to record search timings into (optional)
            match: Document numbers matched in the query (optional)
            event: Analytics event to add counts to (optional)
        Returns:
            List of document dictionaries with content and metadata
        """
        timings = timings if timings is not None else {}
        event = event if event is not None else {}

        # Context search and targeted (number) search run in parallel, so the extra
        # lookup only adds latency if it is slower than the context search.
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Overfetch without a where filter: metadata filtering in ChromaDB
            # scans the whole collection per query (DR-200). Inactive chunks are
            # filtered here instead, then results are trimmed to n_results.
            semantic_future = pool.submit(
                self._timed_query, timings, "semantic_search",
                query_embedding, n_results=self.config.n_results * 2
            )
            targeted_future = pool.submit(self._search_targeted, query_embedding, timings, match, event)
            semantic_results = semantic_future.result()
            targeted_results = targeted_future.result()

        # Targeted chunks first, then context results; de-duplicated by chunk id.
        docs = []
        seen_ids = set()
        targeted_hits = 0
        for results in (targeted_results, semantic_results):
            if not results.get("documents"):
                continue
            for i, doc_text in enumerate(results["documents"][0]):
                if len(docs) >= self.config.n_results:
                    break
                metadata = results["metadatas"][0][i]
                doc_id = results["ids"][0][i]

                if doc_id in seen_ids or not metadata.get("active", True):
                    continue
                seen_ids.add(doc_id)

                if metadata.get("file_name"):
                    metadata["source"] = metadata["file_name"]
                elif doc_id:
                    metadata["source"] = doc_id.split(":")[0]

                docs.append({
                    "page_content": doc_text,
                    "metadata": metadata
                })
                if results is targeted_results:
                    targeted_hits += 1

        event["chunks_from_targeted"] = targeted_hits
        logger.info(f"Found {len(docs)} relevant documents ({targeted_hits} from number/title match)")
        return docs

    def _generate_response(self, docs: List[Dict], query: str, glossary_keys: List[str] = None):
        """
        Generate response using OpenAI based on retrieved documents.

        Args:
            docs: List of document dictionaries
            query: Expanded user query
            glossary_keys: Glossary entries matched in the query (only these go in the prompt)
        Returns:
            (response text, token usage)
        """
        # Build context from documents
        context_text = self._build_context(docs)

        # Add glossary to context
        glossary_text = self.glossary_manager.get_glossary_text(glossary_keys)
        context_with_glossary = f"{glossary_text}\n\n{context_text}"

        # Generate response
        return self.openai_client.generate_chat_response(
            PROMPT_TEMPLATE,
            context=context_with_glossary,
            question=query
        )

    def _build_context(self, docs: List[Dict]) -> str:
        """
        Args:
            docs: List of document dictionaries

        Returns:
            str: Formatted context text
        """
        context_parts = []
        for doc in docs:
            source = doc["metadata"].get("source", "Unknown")
            page = doc["metadata"].get("page_label",
                                      doc["metadata"].get("page", "N/A"))
            content = doc["page_content"]

            context_parts.append(
                f'Document: "{source}"\n'
                f'Page: {page}\n\n'
                f'{content}'
            )

        return "\n\n---\n\n".join(context_parts)
