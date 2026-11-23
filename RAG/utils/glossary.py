import os
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class GlossaryManager:
    
    def __init__(self, config):
        self.config = config
        self.glossary_text = self._load_glossary_text()
        self.glossary_dict = self._load_glossary()
    
    def _load_glossary_text(self) -> str:
        glossary_path = self.config.get_glossary_path()
        logger.info(f"Loading glossary from: {glossary_path}")
        
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Successfully loaded glossary: {len(content)} characters")
                return content
        except FileNotFoundError:
            logger.error(f"Glossary file not found: {glossary_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading glossary file: {str(e)}")
            return ""
    
    def _load_glossary(self) -> Dict[str, str]:
        """Parse glossary text into dictionary"""
        glossary = {}
        
        if not self.glossary_text:
            return glossary
        
        lines = self.glossary_text.strip().splitlines()
        for line in lines:
            if line.startswith("- "):
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    key, value = parts
                    glossary[key.strip()] = value.strip()
        
        logger.info(f"Loaded {len(glossary)} glossary entries")
        return glossary
    
    def expand_acronyms(self, text: str) -> str:
        """Expand acronyms in text"""
        if not self.glossary_dict:
            return text
        
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self.glossary_dict.keys()) + r')\b'
        )
        expanded = pattern.sub(lambda m: self.glossary_dict[m.group(1)], text)
        
        if expanded != text:
            logger.debug("Expanded acronyms in query")
        
        return expanded
    
    def get_glossary_text(self) -> str:
        return self.glossary_text