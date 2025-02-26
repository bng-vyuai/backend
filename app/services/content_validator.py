"""
Content validation service for ensuring consistent and safe AI responses.

This service implements response validation, schema enforcement,
content moderation, and output sanitization for AI model responses.
"""

import re
import json
import os
from typing import Dict, Any, List, Optional, Union, Type
import jsonschema
from pydantic import BaseModel, ValidationError, create_model
from fastapi import HTTPException

from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class ContentValidator:
    """Service for validating and sanitizing AI content."""
    
    def __init__(self):
        """Initialize the content validator service."""
        # Content moderation patterns
        self.moderation_patterns = {
            "personal_data": [
                r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN
                r'\b\d{16}\b',  # Credit card number (simple)
                r'\b\d{13}\b',  # Credit card number (simple)
                r'\b(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'  # Email
            ],
            "offensive_content": [
                # Add patterns for offensive content if needed
                # These would be simple patterns to catch obviously problematic content
            ],
            "unsafe_instructions": [
                r'(?i)\b(?:hack|exploit|bypass|crack)\b.*\b(?:password|security|authentication|firewall)\b',
                r'(?i)\b(?:create|make|build)\b.*\b(?:malware|virus|ransomware|botnet)\b'
            ]
        }
        
        # Common validation schemas
        self.common_schemas = {
            "json_object": {
                "type": "object"
            },
            "json_array": {
                "type": "array"
            },
            "structured_response": {
                "type": "object",
                "required": ["status", "data"],
                "properties": {
                    "status": {"type": "string"},
                    "data": {"type": "object"}
                }
            }
        }
    
    async def validate_response(
        self,
        text: str,
        validation_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and sanitize a model response.
        
        Args:
            text: Response text to validate
            validation_type: Type of validation to perform
            schema: JSON schema for validation
            user_id: Optional user ID for personalized moderation
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "original_text": text,
            "validated_text": text,
            "validation_type": validation_type,
            "is_valid": True,
            "moderation_flags": [],
            "schema_validation": None,
            "modifications": []
        }
        
        # Apply content moderation
        moderation_result = await self.moderate_content(text, user_id=user_id)
        result["moderation_flags"] = moderation_result["flags"]
        
        # Replace any redacted content
        if moderation_result["redacted_text"] != text:
            result["validated_text"] = moderation_result["redacted_text"]
            result["modifications"].append("redacted_sensitive_content")
        
        # Apply schema validation if requested
        if validation_type or schema:
            # Get schema based on type if not provided directly
            validation_schema = schema
            if not validation_schema and validation_type in self.common_schemas:
                validation_schema = self.common_schemas[validation_type]
            
            # Perform validation if we have a schema
            if validation_schema:
                schema_result = await self.validate_schema(result["validated_text"], validation_schema)
                result["schema_validation"] = schema_result
                
                # If schema validation failed but we can fix it, try to do so
                if not schema_result["is_valid"] and schema_result.get("fixed_content"):
                    result["validated_text"] = schema_result["fixed_content"]
                    result["modifications"].append("fixed_schema_errors")
                    # Re-validate to confirm fixes worked
                    schema_result = await self.validate_schema(result["validated_text"], validation_schema)
                    result["schema_validation"] = schema_result
                
                # Update overall validity
                result["is_valid"] = result["is_valid"] and schema_result["is_valid"]
        
        # Apply general text sanitization
        sanitized_text = await self.sanitize_text(result["validated_text"])
        if sanitized_text != result["validated_text"]:
            result["validated_text"] = sanitized_text
            result["modifications"].append("sanitized_text")
        
        return result
    
    async def moderate_content(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check content for policy violations and sensitive information.
        
        Args:
            text: Text to moderate
            user_id: Optional user ID for personalized moderation
            
        Returns:
            Dictionary with moderation results
        """
        result = {
            "original_text": text,
            "redacted_text": text,
            "flags": [],
            "has_violations": False
        }
        
        # Check each pattern category
        for category, patterns in self.moderation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Found a match
                    result["flags"].append({
                        "category": category,
                        "match": match.group(),
                        "position": match.span()
                    })
                    result["has_violations"] = True
                    
                    # Redact the content based on category
                    if category == "personal_data":
                        # Redact personal data with asterisks
                        redacted = '*' * len(match.group())
                        result["redacted_text"] = result["redacted_text"][:match.start()] + redacted + result["redacted_text"][match.end():]
        
        return result
    
    async def validate_schema(
        self,
        text: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate text against a JSON schema.
        
        Args:
            text: Text to validate
            schema: JSON schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": False,
            "errors": [],
            "parsed_content": None,
            "fixed_content": None
        }
        
        # Try to parse as JSON if schema validation is requested
        try:
            # Extract JSON from the text (in case it's embedded in other text)
            json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            
            if json_match:
                # Found JSON in code blocks
                json_str = json_match.group(1)
            else:
                # Try to find JSON-like content with curly braces
                json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Assume the whole text is JSON
                    json_str = text
            
            # Parse the extracted JSON
            parsed = json.loads(json_str)
            result["parsed_content"] = parsed
            
            # Validate against schema
            jsonschema.validate(parsed, schema)
            
            # If we got here, validation succeeded
            result["is_valid"] = True
            
        except json.JSONDecodeError as e:
            result["errors"].append(f"JSON decode error: {str(e)}")
            # Try to fix common JSON errors
            fixed_str = self._try_fix_json(json_str)
            if fixed_str != json_str:
                try:
                    fixed_parsed = json.loads(fixed_str)
                    result["fixed_content"] = fixed_str
                    result["parsed_content"] = fixed_parsed
                    
                    # Validate fixed content
                    try:
                        jsonschema.validate(fixed_parsed, schema)
                        result["is_valid"] = True
                    except jsonschema.exceptions.ValidationError as ve:
                        result["errors"].append(f"Schema validation error after fixing: {str(ve)}")
                except:
                    # Still invalid after fixes
                    pass
        
        except jsonschema.exceptions.ValidationError as e:
            result["errors"].append(f"Schema validation error: {str(e)}")
            # Try to fix schema errors if possible
            if isinstance(result["parsed_content"], dict):
                fixed_content = self._try_fix_schema_issues(result["parsed_content"], schema)
                if fixed_content != result["parsed_content"]:
                    result["fixed_content"] = json.dumps(fixed_content)
        
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
        
        return result
    
    async def sanitize_text(self, text: str) -> str:
        """
        Sanitize text for common issues and formatting problems.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix common formatting issues
        
        # 1. Fix markdown code blocks (ensure they have triple backticks)
        unclosed_blocks = text.count("```") % 2
        if unclosed_blocks:
            text += "\n```"
        
        # 2. Fix incomplete sentences at the end
        if text and text[-1] not in ".!?;:\"')]}":
            if not text.endswith("..."):
                text += "..."
        
        # 3. Fix common HTML issues (unclosed tags)
        common_tags = ["p", "div", "span", "strong", "em", "ul", "ol", "li", "table", "tr", "td", "th"]
        for tag in common_tags:
            # Check for unclosed tags
            open_count = len(re.findall(f"<{tag}[^>]*>", text, re.IGNORECASE))
            close_count = len(re.findall(f"</{tag}>", text, re.IGNORECASE))
            
            # Add closing tags if needed
            if open_count > close_count:
                for _ in range(open_count - close_count):
                    text += f"</{tag}>"
        
        return text
    
    def _try_fix_json(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting errors.
        
        Args:
            json_str: JSON string to fix
            
        Returns:
            Fixed JSON string
        """
        # Make a copy to work on
        fixed = json_str
        
        # Fix 1: Missing quotes around property names
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
        
        # Fix 2: Single quotes instead of double quotes
        # Only fix single quotes not inside double quotes
        fixed = re.sub(r"(?<!\\)'([^']*?)(?<!\\)'", r'"\1"', fixed)
        
        # Fix 3: Trailing commas in arrays/objects
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # Fix 4: Add missing braces/brackets if they appear unbalanced
        open_curly = fixed.count('{')
        close_curly = fixed.count('}')
        if open_curly > close_curly:
            fixed += '}' * (open_curly - close_curly)
        
        open_square = fixed.count('[')
        close_square = fixed.count(']')
        if open_square > close_square:
            fixed += ']' * (open_square - close_square)
        
        return fixed
    
    def _try_fix_schema_issues(self, content: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix common schema validation issues.
        
        Args:
            content: Parsed content
            schema: JSON schema
            
        Returns:
            Fixed content dict
        """
        # Make a copy to avoid modifying the original
        fixed = content.copy()
        
        # Fix 1: Add required properties with default values
        if 'required' in schema and 'properties' in schema:
            for prop in schema['required']:
                if prop not in fixed and prop in schema['properties']:
                    prop_schema = schema['properties'][prop]
                    # Add with default value based on type
                    if prop_schema.get('type') == 'string':
                        fixed[prop] = ""
                    elif prop_schema.get('type') == 'number':
                        fixed[prop] = 0
                    elif prop_schema.get('type') == 'boolean':
                        fixed[prop] = False
                    elif prop_schema.get('type') == 'array':
                        fixed[prop] = []
                    elif prop_schema.get('type') == 'object':
                        fixed[prop] = {}
        
        # Fix 2: Handle type mismatches for basic types
        if 'properties' in schema:
            for prop, prop_schema in schema['properties'].items():
                if prop in fixed and 'type' in prop_schema:
                    expected_type = prop_schema['type']
                    
                    # Type conversions
                    if expected_type == 'string' and not isinstance(fixed[prop], str):
                        fixed[prop] = str(fixed[prop])
                    elif expected_type == 'number' and isinstance(fixed[prop], str):
                        try:
                            fixed[prop] = float(fixed[prop])
                        except:
                            pass
                    elif expected_type == 'integer' and isinstance(fixed[prop], str):
                        try:
                            fixed[prop] = int(fixed[prop])
                        except:
                            pass
                    elif expected_type == 'boolean' and isinstance(fixed[prop], str):
                        fixed[prop] = fixed[prop].lower() in ('true', 'yes', '1')
        
        return fixed


# Singleton instance
content_validator = ContentValidator()