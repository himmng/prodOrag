"""Multi-query expansion retriever.

LLM-generates N alternative phrasings, retrieves each, fuses with RRF.
The expansion prompt is loaded from prompts/templates/query_expansion.yaml.

Phase 2 finding: helps qualitatively on hard compound queries but per-variant
reranker filtering can spike aggregate refusal rate. Use inside the Phase 3
router on SELECTED queries — not as a default for everything.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from rag_pipeline.config import log
from rag_pipeline.prompts import render_prompt
from rag_pipeline.retrievers.base import BaseRetriever, reciprocal_rank_fusion

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel


class MultiQueryRetriever(BaseRetriever):
    """Expand → retrieve each --> RRF-fuse. Wraps any base retriever."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: "BaseChatModel",
        n_variants: int = 3,
        rrf_k: int = 60,
        per_variant_fetch: int = 10,
        prompt_name: str = "query_expansion",   # ← name of the YAML prompt
    ):
        self.base = base_retriever
        self.llm = llm
        self.n_variants = n_variants
        self.rrf_k = rrf_k
        self.per_variant_fetch = per_variant_fetch
        self.prompt_name = prompt_name

    def expand(self, query: str) -> list[str]:
        """LLM-generate variants. Always returns [original, ...up to N variants]."""
        prompt = render_prompt(self.prompt_name, n=self.n_variants, query=query)
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            cleaned = re.sub(
                r"^```(?:json)?|```$", "",
                resp.content.strip(),
                flags=re.MULTILINE,
            ).strip()
            data = json.loads(cleaned)
            variants = [v.strip() for v in data.get("variants", []) if v and isinstance(v, str)]
        except Exception as e:
            log.warning(f"[MQ] expansion failed ({e}); falling back to single query")
            variants = []

        seen = {query.strip().lower()}
        out = [query]
        for v in variants:
            if v.strip().lower() not in seen:
                out.append(v)
                seen.add(v.strip().lower())
        return out[: self.n_variants + 1]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple["Document", float]]:
        variants = self.expand(query)
        log.info(f"[MQ] {len(variants)} variants for {query[:60]!r}")

        rankings: list[list["Document"]] = []
        for v in variants:
            results = self.base.retrieve(v, top_k=self.per_variant_fetch)
            rankings.append([d for d, _ in results])

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        return fused[: top_k]