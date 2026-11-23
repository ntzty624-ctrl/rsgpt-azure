import logging
from typing import Dict, List, Any
from ..config import Config
from ..utils.glossary import GlossaryManager
from ..utils.document_linker import DocumentLinker
from ..clients.openai_client import OpenAIClient
from ..clients.chromadb_client import ChromaDBClient
from ..prompts import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


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
        
        try:
            # Expand acronyms in query
            expanded_query = self.glossary_manager.expand_acronyms(query_text)
            logger.debug(f"Expanded query: {expanded_query}")
            
            # Generate embedding for query
            query_embedding = self.openai_client.generate_embedding(expanded_query)
            
            # Search for relevant documents
            docs = self._search_documents(query_embedding)
            
            if not docs:
                return "No relevant documents found."
            
            # Generate response
            response_text = self._generate_response(docs, expanded_query)
            
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
            raise
    
    def _search_documents(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Args:
            query_embedding: Query embedding vector
        Returns:
            List of document dictionaries with content and metadata
        """
        # Overfetch without a where filter: metadata filtering in ChromaDB
        # scans the whole collection per query (DR-200). Inactive chunks are
        # filtered here instead, then results are trimmed to n_results.
        results = self.chromadb_client.query_collection(
            query_embedding,
            n_results=self.config.n_results * 2
        )

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