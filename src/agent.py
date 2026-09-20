from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn


    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        if self.store.get_collection_size() == 0:
            return "Kho tài liệu hiện đang trống. Không tìm thấy thông tin phù hợp."
        chunks = self.store.search(question, top_k=top_k)
        if not chunks: return "Không tìm thấy thông tin phù hợp trong kho tài liệu."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source") or meta.get("doc_id") or chunk.get("id")
            source_str = f" (Nguồn: {source})" if source else ""
            content = chunk.get("content", "").strip()
            context_parts.append(f"[{i}]{source_str}\n{content}")
        context = "\n\n".join(context_parts)

        prompt = (
            "Bạn là trợ lý giải đáp thắc mắc dựa trên tài liệu được cung cấp.\n"
            "Chỉ sử dụng thông tin trong ngữ cảnh dưới đây để trả lời câu hỏi. "
            "Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy.\n"
            "Hãy trích dẫn số thứ tự của tài liệu (ví dụ: [1], [2]) tương ứng khi đưa ra thông tin.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)
