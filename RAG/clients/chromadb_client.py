import logging
from typing import List, Dict, Any
from chromadb import HttpClient
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """Client for ChromaDB operations."""
    
    def __init__(self, config):
        """Initialize ChromaDB client with configuration."""
        self.config = config
        self.client = HttpClient(
            host=config.chromadb_host,
            port=config.chromadb_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=config.chromadb_token
            )
        )
        self.collection = self.client.get_collection(name=config.chromadb_collection)
    
    def query_collection(self, query_embedding: List[float], n_results: int = None) -> Dict[str, Any]:
        """     
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return        
        Returns:
            Dict containing query results
        """
        n_results = n_results or self.config.n_results
        
        logger.debug(f"Querying ChromaDB for {n_results} results")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents"]
        )
        
        return results