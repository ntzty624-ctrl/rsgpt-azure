import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from ..config import Config
from ..utils.glossary import GlossaryManager
from ..utils.document_linker import DocumentLinker
from ..clients.openai_client import OpenAIClient
from ..clients.chromadb_client import ChromaDBClient
from ..prompts import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Same patterns as rs-gpt sync_compare._DOC_NUMBER_RE (SIN-A01-002, T-TI-WO-5958, S123),
# but searched anywhere in the query instead of anchored to the start of a file name.
DOC_NUMBER_RE = re.compile(r"\b(SIN-[A-Z]\d{2}-\d{3}|[A-Z]-[A-Z]{2}-[A-Z]{2}-\d+|S\d{3,4})\b", re.IGNORECASE)


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
    
    def process_query(self, query_text: str) -> str:
        """
        Process a user query and return formatted response with document links.
        Args:
            query_text: User's query
        Returns:
            str: Formatted response with document suggestions and links
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        timings = {}
        t_total = time.perf_counter()
        
        try:
            # Detect document numbers on the raw query (before glossary expansion can alter them)
            doc_numbers = self._extract_document_numbers(query_text)

            # Expand acronyms in query
            expanded_query = self.glossary_manager.expand_acronyms(query_text)
            logger.debug(f"Expanded query: {expanded_query}")
            
            # Generate embedding for query
            t = time.perf_counter()
            query_embedding = self.openai_client.generate_embedding(expanded_query)
            timings["embed_ms"] = self._ms(t)
            
            # Search for relevant documents
            t = time.perf_counter()
            docs = self._search_documents(query_embedding, doc_numbers, timings)
            timings["search_total_ms"] = self._ms(t)
            
            if not docs:
                return "No relevant documents found."
            
            # Generate response
            t = time.perf_counter()
            response_text = self._generate_response(docs, expanded_query)
            timings["llm_ms"] = self._ms(t)
            
            # Build metadata lookup for document linker
            doc_metadata = {}
            for doc in docs:
                source = doc["metadata"].get("source", "")
                if source and source not in doc_metadata:
                    doc_metadata[source] = doc["metadata"]
            
            # Add SharePoint links
            final_response = self.document_linker.add_links_to_response(response_text, doc_metadata)

            timings["total_ms"] = self._ms(t_total)
            logger.info(f"Timings: {timings} doc_numbers={doc_numbers}")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _ms(start: float) -> int:
        return round((time.perf_counter() - start) * 1000)

    @staticmethod
    def _extract_document_numbers(query_text: str) -> List[str]:
        """Return document numbers mentioned in the query, as typed and upper-cased."""
        found = []
        for m in DOC_NUMBER_RE.findall(query_text):
            for variant in (m, m.upper()):
                if variant not in found:
                    found.append(variant)
        return found

    def _timed_query(self, timings: Dict[str, int], key: str, *args, **kwargs) -> Dict[str, Any]:
        t = time.perf_counter()
        try:
            return self.chromadb_client.query_collection(*args, **kwargs)
        finally:
            timings[key] = self._ms(t)

    def _search_documents(self, query_embedding: List[float], doc_numbers: List[str] = None,
                          timings: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Args:
            query_embedding: Query embedding vector
            doc_numbers: Document numbers detected in the query (optional)
            timings: Dict to record per-search timings into (optional)
        Returns:
            List of document dictionaries with content and metadata
        """
        timings = timings if timings is not None else {}

        # Run the context search and the document-number search in parallel, so the
        # extra lookup only adds latency if it is slower than the context search.
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Overfetch without a where filter: metadata filtering in ChromaDB
            # scans the whole collection per query (DR-200). Inactive chunks are
            # filtered here instead, then results are trimmed to n_results.
            semantic_future = pool.submit(
                self._timed_query, timings, "semantic_search_ms",
                query_embedding, n_results=self.config.n_results * 2
            )
            # Most relevant chunks *within* the documents named in the query.
            doc_future = None
            if doc_numbers:
                doc_future = pool.submit(
                    self._timed_query, timings, "doc_number_search_ms",
                    query_embedding, n_results=self.config.doc_number_n_results,
                    where={"$and": [{"document_number": {"$in": doc_numbers}}, {"active": True}]}
                )
            semantic_results = semantic_future.result()
            doc_results = {}
            if doc_future:
                try:
                    doc_results = doc_future.result()
                except Exception as e:
                    # Never fail the request because of the extra lookup
                    logger.warning(f"Document number search failed: {e}", exc_info=True)

        # Named-document chunks first, then context results; de-duplicated by chunk id.
        docs = []
        seen_ids = set()
        doc_number_hits = 0
        for results in (doc_results, semantic_results):
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
                if results is doc_results:
                    doc_number_hits += 1
        
        logger.info(f"Found {len(docs)} relevant documents ({doc_number_hits} from document number match)")
        return docs
    
    def _generate_response(self, docs: List[Dict], query: str) -> str:
        """
        Generate response using OpenAI based on retrieved documents.
        
        Args:
            docs: List of document dictionaries
            query: Expanded user query    
        Returns:
            str: Generated response text
        """
        # Build context from documents
        context_text = self._build_context(docs)
        
        # Add glossary to context
        glossary_text = self.glossary_manager.get_glossary_text()
        context_with_glossary = f"{glossary_text}\n\n{context_text}"
        
        # Generate response
        response = self.openai_client.generate_chat_response(
            PROMPT_TEMPLATE,
            context=context_with_glossary,
            question=query
        )
        
        return response
    
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