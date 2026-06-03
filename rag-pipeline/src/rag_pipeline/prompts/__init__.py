"""Prompt management -- version controlled, externalized prompts."""

from rag_pipeline.prompts.loader import (
    PromptManager,
    get_prompt,
    render_prompt,
)

__all__ = ["PromptManager", "get_prompt", "render_prompt"]