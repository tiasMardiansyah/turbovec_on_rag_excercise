from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly "
    "on the provided context. If the context does not contain enough "
    "information to answer the question, say so clearly. Always cite "
    "which part of the context supports your answer."
)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Generate an answer using RAG: question + retrieved context chunks."""
    llm = ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
        max_tokens=1024,
    )

    context_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Context:\n{context_text}\n\n"
                    f"---\n\n"
                    f"Question: {question}\n\n"
                    f"Answer based only on the context above:"
                )
            ),
        ]
    )
    return response.content or ""
