import logging
import os
import azure.functions as func
from dotenv import load_dotenv
from .services.rag_service import RAGService
from .config import Config

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Singleton 
_rag_service = None
_config = None

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger endpoint for document query processing.
    Requires function key authentication.
    Args:
        req: HTTP request with query_text parameter
        
    Returns:
        func.HttpResponse: Formatted response with document suggestions and feedback link
    """
    logger.info('Azure Function triggered.')
    
    try:
        # Extract query text from request
        query_text = _extract_query_text(req)
        
        if not query_text:
            return func.HttpResponse(
                "Please pass a query_text on the query string or in the request body",
                status_code=400
            )
        
        # Initialize service and process query
        global _rag_service, _config
        
        if _config is None:
            logger.info("Initializing Config singleton")
            _config = Config()
            logger.info("Config singleton initialized successfully")
        
        if _rag_service is None:
            logger.info("Initializing RAG service singleton")
            _rag_service = RAGService(_config)
            logger.info("RAG service singleton initialized successfully")
        
        # Process the query
        result = _rag_service.process_query(query_text)
        
        return func.HttpResponse(f"\n{result}")
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        logger.warning("Error detected - resetting singletons for next request")
        
        # Reset singletons on error to allow recovery
        _rag_service = None
        _config = None
        
        return func.HttpResponse(
            f"An error occurred while processing your request: {str(e)}",
            status_code=500
        )


def _extract_query_text(req: func.HttpRequest) -> str:
    """
    Extract query_text from request parameters or body.
    
    Args:
        req: HTTP request object
        
    Returns:
        str: Extracted query text or None
    """
    query_text = req.params.get('query_text')
    
    if not query_text:
        try:
            req_body = req.get_json()
            query_text = req_body.get('query_text') if req_body else None
        except (ValueError, AttributeError):
            pass
    
    return query_text