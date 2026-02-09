"""
LLM (Large Language Model) manager for the RAG pipeline.
Uses Nebius API for fast inference (Meta-Llama-3.1-8B-Instruct).

To swap LLM models:
1. Change LLM_MODEL_NAME in config.py
2. Update the prompt template in rag_chain.py if needed
"""

from typing import Optional, List, Dict, Any
from openai import OpenAI

from ..config import settings


class NebiusLLM:
    """
    LLM class using Nebius API (OpenAI-compatible).
    """

    def __init__(self):
        """Initialize Nebius client."""
        if not settings.NEBIUS_API_KEY:
            raise ValueError(
                "NEBIUS_API_KEY not set. Please add it to your .env file."
            )

        self.client = OpenAI(
            base_url=settings.NEBIUS_BASE_URL,
            api_key=settings.NEBIUS_API_KEY,
        )
        self.model = settings.LLM_MODEL_NAME

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user's prompt/question with context.
            system_prompt: Optional system instructions.

        Returns:
            Generated text response.
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

        return response.choices[0].message.content

    def invoke(self, prompt: str) -> str:
        """
        LangChain-compatible invoke method.

        Args:
            prompt: The full prompt string.

        Returns:
            Generated response.
        """
        return self.generate(prompt)


class LLMManager:
    """
    Manages the LLM lifecycle.
    Singleton pattern ensures only one client instance.
    """

    _instance: Optional["LLMManager"] = None
    _llm: Optional[NebiusLLM] = None

    def __new__(cls) -> "LLMManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize LLM manager (only runs once due to singleton)."""
        pass  # Lazy loading - connects on first access

    def _load_model(self) -> None:
        """Connect to Nebius API."""
        print(f"[LLM] Connecting to Nebius API...")
        print(f"[LLM] Model: {settings.LLM_MODEL_NAME}")

        self._llm = NebiusLLM()

        print("[LLM] Connected successfully")

    @property
    def llm(self) -> NebiusLLM:
        """
        Get the LLM instance (lazy loading).

        Returns:
            NebiusLLM: The LLM client.
        """
        if self._llm is None:
            self._load_model()
        return self._llm

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt.

        Returns:
            Generated text response.
        """
        return self.llm.generate(prompt)

    def is_loaded(self) -> bool:
        """Check if the client is connected."""
        return self._llm is not None


# Global accessor function
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """
    Get or create the LLM manager instance.
    Use this instead of direct instantiation.
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
