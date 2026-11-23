import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    
    # Azure Config
    azure_openai_key: str = None
    azure_openai_endpoint: str = "https://iv-aue-de-openai.openai.azure.com"
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_embedding_deployment: str = None
    api_version: str = "2024-12-01-preview"
    
    #ChromaDB Config
    chromadb_host: str = None
    chromadb_port: int = None
    chromadb_token: str = None
    chromadb_collection: str = "documents"
    
    # File paths - Mounted file share
    base_dir: str = None
    mount_path: str = None
    glossary_file: str = "glossary.txt"
    
    # SharePoint 
    sharepoint_prefix: str = "https://kiwirail.sharepoint.com/"
    
    # Query settings
    n_results: int = 30
    doc_number_n_results: int = 10  # max chunks taken from documents named in the query
    max_tokens: int = 5000
    temperature: float = 0.7 # Lower the more accurate
    
    def __post_init__(self):
        
        self.azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_embedding_deployment = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT", 
            "text-embedding-ada-002"
        )
        self.chromadb_host = os.getenv("CHROMADB_HOST")
        self.chromadb_port = int(os.getenv("CHROMADB_PORT", 8000))
        self.chromadb_token = os.getenv("CHROMADB_TOKEN", "")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Mount path configuration - uses /mnt/fileshare from Azure file share mount
        self.mount_path = "/mnt/fileshare"
    
    def get_glossary_path(self) -> str:
        """Get full path to glossary file on mounted share"""
        return os.path.join(self.mount_path, "azurefiles", self.glossary_file)