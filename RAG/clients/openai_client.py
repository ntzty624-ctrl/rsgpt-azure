import logging
import openai
from typing import Dict, List, Tuple
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for Azure OpenAI operations."""
    
    def __init__(self, config):
        """Initialize OpenAI client with configuration."""
        self.config = config
        self.client = openai.AzureOpenAI(
            api_key=config.azure_openai_key,
            api_version=config.api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Args:
            text: Input text            
        Returns:
            List of embedding values            
        Raises:
            ValueError: If no embedding is returned
        """
        logger.debug(f"Generating embedding for text: {text[:100]}...")
        
        response = self.client.embeddings.create(
            model=self.config.azure_embedding_deployment,
            input=text
        )
        
        if not response.data:
            raise ValueError("No embedding returned from OpenAI")
        
        return response.data[0].embedding
    
    def generate_chat_response(self, template: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """   
        Args:
            template: Prompt template string
            **kwargs: Template variables           
        Returns:
            (response text, token usage {"prompt_tokens", "completion_tokens"})
        """
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(**kwargs)
        
        response = self.client.chat.completions.create(
            model=self.config.azure_openai_deployment,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a document assistant. Always follow formatting "
                              "and do not invent titles. Only use the filenames given "
                              "in the Document: \"...\" blocks."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        usage = getattr(response, "usage", None)
        tokens = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return response.choices[0].message.content, tokens
