import logging
import time
import uuid
from typing import Dict, List, Any
from ..config import Config
from ..utils.glossary import GlossaryManager
from ..utils.document_linker import DocumentLinker
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
        stage = "glossary"
        
        try:
            # Expand acronyms in query (reloads the glossary first if the file changed)
            t = time.perf_counter()
            self.glossary_manager.reload_if_changed()
            expanded_query, glossary_keys = self.glossary_manager.expand_acronyms(query_text)
            timings["glossary"] = _ms(t)
            event["glossary_matched"] = len(glossary_keys)
            logger.debug(f"Expanded query: {expanded_query}")
            
            # Generate embedding for query
            stage = "embed"
            t = time.perf_counter()
            query_embedding = self.openai_client.generate_embedding(expanded_query)
            timings["embed"] = _ms(t)
            
            # Search for relevant documents
            stage = "search"
            t = time.perf_counter()
            docs = self._search_documents(query_embedding, timings)
            timings["search_total"] = _ms(t)
            event["chunks_retrieved"] = len(docs)
            event["docs_in_context"] = len({d["metadata"].get("source") for d in docs})
            
            if not docs:
                event["route"] = "no_results"
                return "No relevant documents found."
            
            # Generate response
            stage = "llm"
            t = time.perf_counter()
            response_text, tokens = self._generate_response(docs, expanded_query, glossary_keys)
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
    
    def _search_documents(self, query_embedding: List[float], timings: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Args:
            query_embedding: Query embedding vector
            timings: Dict to record search timings into (optional)
        Returns:
            List of document dictionaries with content and metadata
        """
        # Overfetch without a where filter: metadata filtering in ChromaDB
        # scans the whole collection per query (DR-200). Inactive chunks are
        # filtered here instead, then results are trimmed to n_results.
        t = time.perf_counter()
        results = self.chromadb_client.query_collection(
            query_embedding,
            n_results=self.config.n_results * 2
        )
        if timings is not None:
            timings["semantic_search"] = _ms(t)

        docs = []
        if results.get("documents"):
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                doc_id = results["ids"][0][i]

                if not metadata.get("active", True):
                    continue

                if metadata.get("file_name"):
                    metadata["source"] = metadata["file_name"]
                elif doc_id:
                    metadata["source"] = doc_id.split(":")[0]
                
                docs.append({
                    "page_content": doc_text,
                    "metadata": metadata
                })

                if len(docs) >= self.config.n_results:
                    break
        
        logger.info(f"Found {len(docs)} relevant documents")
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