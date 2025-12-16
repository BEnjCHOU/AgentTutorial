"""
Context Evaluation Module
Evaluates the quality and relevance of context returned by the agent.
"""
from typing import Dict, List, Any
from llama_index.llms.openai import OpenAI
import json
import re


class ContextEvaluator:
    """Evaluates context quality, relevance, and completeness."""
    
    def __init__(self, llm=None):
        self.llm = llm or OpenAI(model="gpt-4o-mini")
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            return {}
    
    async def evaluate_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluate how relevant the context is to the query.
        Returns a score from 0-1 and explanation.
        """
        prompt = f"""Evaluate the relevance of the following context to the query.

Query: {query}

Context: {context}

Rate the relevance on a scale of 0.0 to 1.0, where:
- 1.0 = Highly relevant, directly answers the query
- 0.5 = Somewhat relevant, partially addresses the query
- 0.0 = Not relevant, does not address the query

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "explanation": "<brief explanation of the score>",
    "key_points": ["<list of key points from context>"]
}}
"""
        try:
            response = await self.llm.acomplete(prompt)
            result = self._extract_json(str(response))
            return {
                "relevance_score": float(result.get("score", 0.0)),
                "explanation": result.get("explanation", ""),
                "key_points": result.get("key_points", [])
            }
        except Exception as e:
            print(f"Error evaluating relevance: {e}")
            return {
                "relevance_score": 0.0,
                "explanation": f"Evaluation error: {str(e)}",
                "key_points": []
            }
    
    async def evaluate_completeness(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluate how complete the context is in answering the query.
        """
        prompt = f"""Evaluate the completeness of the following context in answering the query.

Query: {query}

Context: {context}

Rate the completeness on a scale of 0.0 to 1.0, where:
- 1.0 = Complete answer, all aspects covered
- 0.5 = Partial answer, some aspects missing
- 0.0 = Incomplete answer, major aspects missing

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "explanation": "<brief explanation>",
    "missing_aspects": ["<list of missing aspects if any>"]
}}
"""
        try:
            response = await self.llm.acomplete(prompt)
            result = self._extract_json(str(response))
            return {
                "completeness_score": float(result.get("score", 0.0)),
                "explanation": result.get("explanation", ""),
                "missing_aspects": result.get("missing_aspects", [])
            }
        except Exception as e:
            print(f"Error evaluating completeness: {e}")
            return {
                "completeness_score": 0.0,
                "explanation": f"Evaluation error: {str(e)}",
                "missing_aspects": []
            }
    
    async def evaluate_quality(self, query: str, context: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of context quality.
        Returns relevance, completeness, and overall quality metrics.
        """
        relevance = await self.evaluate_relevance(query, context)
        completeness = await self.evaluate_completeness(query, context)
        
        # Calculate overall quality score (weighted average)
        overall_score = (relevance["relevance_score"] * 0.6 + 
                        completeness["completeness_score"] * 0.4)
        
        return {
            "overall_quality_score": overall_score,
            "relevance": relevance,
            "completeness": completeness,
            "recommendation": self._get_recommendation(overall_score)
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on quality score."""
        if score >= 0.8:
            return "High quality context - suitable for use"
        elif score >= 0.6:
            return "Moderate quality - may need additional context"
        elif score >= 0.4:
            return "Low quality - consider refining query or adding more sources"
        else:
            return "Poor quality - context may not be relevant or complete"

