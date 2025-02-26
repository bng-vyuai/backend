"""
Enhancer service to improve consistency and quality of AI responses.

This service applies post-processing enhancements to ensure consistent
formatting and quality across different AI providers.
"""

from typing import Dict, Any, List, Optional
import re

from app.core.models_config import get_model_config
from app.utils.logger import logger


class EnhancerService:
    """Service for enhancing and standardizing AI model outputs."""
    
    def __init__(self):
        """Initialize the enhancer service."""
        pass
    
    async def enhance_response(
        self,
        text: str,
        model: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Apply enhancements to the model's response.
        
        Args:
            text: The raw text response from the model
            model: The model that generated the response
            messages: The conversation messages
            
        Returns:
            Enhanced text response
        """
        # Apply a series of enhancements
        enhanced_text = text
        
        # 1. Fix markdown formatting issues
        enhanced_text = self._fix_markdown_formatting(enhanced_text)
        
        # 2. Fix code blocks
        enhanced_text = self._fix_code_blocks(enhanced_text)
        
        # 3. Fix truncated sentences (if they appear cut off)
        enhanced_text = self._fix_truncated_sentences(enhanced_text)
        
        # 4. Remove any model self-references or signatures
        enhanced_text = self._remove_model_signatures(enhanced_text, model)
        
        # 5. Fix list formatting
        enhanced_text = self._fix_list_formatting(enhanced_text)
        
        # 6. Fix table formatting
        enhanced_text = self._fix_table_formatting(enhanced_text)
        
        # Log enhancement metrics
        if enhanced_text != text:
            logger.info(f"Enhanced response for model {model} ({len(text)} -> {len(enhanced_text)} chars)")
        
        return enhanced_text
    
    def _fix_markdown_formatting(self, text: str) -> str:
        """Fix common markdown formatting issues."""
        # Fix headings without space after #
        text = re.sub(r'(?m)^(#+)([^ \n])', r'\1 \2', text)
        
        # Ensure proper spacing around bold/italic markers
        text = re.sub(r'(\w)(\*\*|\*)(\w)', r'\1 \2\3', text)
        text = re.sub(r'(\w)(__|\*)(\w)', r'\1 \2\3', text)
        
        # Fix blockquotes without space after >
        text = re.sub(r'(?m)^(>+)([^ \n])', r'\1 \2', text)
        
        return text
    
    def _fix_code_blocks(self, text: str) -> str:
        """Fix common code block formatting issues."""
        # Ensure code blocks have proper syntax highlighting tag
        # Find code blocks with no language specified
        blocks = re.finditer(r'```\s*\n', text)
        
        # Create a new text with fixed code blocks
        last_end = 0
        new_text = ""
        
        for match in blocks:
            # Add text before this code block
            new_text += text[last_end:match.start()]
            # Replace with code block with generic language
            new_text += "```text\n"
            last_end = match.end()
        
        # Add the rest of the text
        new_text += text[last_end:]
        
        if new_text:
            text = new_text
        
        # Ensure code blocks are closed
        open_blocks = text.count("```") % 2
        if open_blocks:
            text += "\n```"
        
        return text
    
    def _fix_truncated_sentences(self, text: str) -> str:
        """Attempt to fix sentences that appear truncated."""
        # Check if text ends with an incomplete sentence
        if text and text[-1] not in ".!?;:\"')}]":
            # If ends without punctuation, add an ellipsis to indicate truncation
            if not text.endswith("..."):
                text += "..."
        
        return text
    
    def _remove_model_signatures(self, text: str, model: str) -> str:
        """Remove model signatures or self-references."""
        # Remove "As [Model Name]," prefixes
        model_config = get_model_config(model)
        if model_config:
            provider = model_config.provider
            
            # Common signature patterns by provider
            if provider == "openai":
                text = re.sub(r'(?i)^As an AI language model,? ', '', text)
                text = re.sub(r'(?i)^As GPT[^,.]*, ', '', text)
            elif provider == "anthropic":
                text = re.sub(r'(?i)^As Claude,? ', '', text)
                text = re.sub(r'(?i)^As an AI assistant,? ', '', text)
            elif provider == "deepseek":
                text = re.sub(r'(?i)^As DeepSeek[^,.]*, ', '', text)
                text = re.sub(r'(?i)^As an AI model,? ', '', text)
            
            # General patterns
            text = re.sub(r'(?i)^As an AI,? ', '', text)
            
        return text
    
    def _fix_list_formatting(self, text: str) -> str:
        """Fix common list formatting issues."""
        # Fix numbered lists with no space after number
        text = re.sub(r'(?m)^(\d+\.)([^ \n])', r'\1 \2', text)
        
        # Fix bullet lists with no space after marker
        text = re.sub(r'(?m)^([*-])([^ \n])', r'\1 \2', text)
        
        return text
    
    def _fix_table_formatting(self, text: str) -> str:
        """Fix common markdown table formatting issues."""
        # Find markdown tables
        table_pattern = r'(?m)^(\|[^\n]+\|)$\s*^(\|[\s-:]+\|)$'
        tables = re.finditer(table_pattern, text)
        
        # Process each table
        last_end = 0
        new_text = ""
        
        for match in tables:
            # Add text before this table
            new_text += text[last_end:match.start()]
            
            # Get header and separator rows
            header = match.group(1)
            separator = match.group(2)
            
            # Fix separator row to ensure it has correct format
            # Each cell should have at least 3 dashes
            fixed_separator = re.sub(r'\|[\s-:]+', lambda m: m.group(0).replace('-', '---') if '-' in m.group(0) else m.group(0), separator)
            
            # Replace with fixed table
            new_text += header + "\n" + fixed_separator
            last_end = match.end()
        
        # Add the rest of the text
        new_text += text[last_end:]
        
        if new_text:
            text = new_text
        
        return text