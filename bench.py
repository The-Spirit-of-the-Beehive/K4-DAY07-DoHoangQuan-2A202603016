from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.chunking import FixedSizeChunker, RecursiveChunker
from src.embeddings import LocalEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

# =====================================================================
# 1. ĐỊNH NGHĨA HEADING CHUNKER (Chiến lược 3 của Hoàng theo yêu cầu)
# =====================================================================
class HeadingChunker:
    """Tách văn bản theo các tiêu đề Markdown (#, ##, ###).
    Nếu section dài hơn chunk_size, fallback sang RecursiveChunker
    và gắn lại tiêu đề vào từng mảnh con để giữ ngữ cảnh.
    """

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size
        self.fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        lines = text.splitlines()
        sections = []
        current_header = ""
        current_lines = []

        for line in lines:
            if re.match(r"^#{1,6}\s+", line):
                if current_lines:
                    sections.append((current_header, "\n".join(current_lines).strip()))
                current_header = line
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_header, "\n".join(current_lines).strip()))

        chunks = []
        for header, sec_text in sections:
            if not sec_text:
                continue
            if len(sec_text) <= self.chunk_size:
                chunks.append(sec_text)
            else:
                sub_chunks = self.fallback.chunk(sec_text)
                for sc in sub_chunks:
                    # Gắn lại tiêu đề mục nếu mảnh con chưa có
                    if header and not sc.startswith("#"):
                        chunks.append(f"{header}\n{sc}")
                    else:
                        chunks.append(sc)
        return chunks


# =====================================================================
# 2. HÀM ĐỌC VÀ TÁCH FRONTMATTER
# =====================================================================
def parse_markdown_file(file_path: Path) -> tuple[dict[str, Any], str]:
    """Tách metadata YAML frontmatter và content của file Markdown."""
    text = file_path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            content = parts[2].strip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip("\"'")

    return metadata, content


# =====================================================================
# 3. 5 BENCHMARK QUERIES ĐÃ THỐNG NHẤT CỦA NHÓM
# =====================================================================
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người mua có bao nhiêu ngày để yêu cầu trả hàng hoàn tiền sau khi giao hàng thành công?",
        "filter": {"audience": "buyer"},
        "gold_doc": "shopee-dam-bao",
        "gold_answer": "15 ngày kể từ khi đơn hàng cập nhật giao hàng thành công.",
    },
    {
        "id": 2,
        "query": "Khi Shopee đang xem xét yêu cầu trả hàng hoàn tiền thì bao lâu có kết quả?",
        "filter": None,
        "gold_doc": "quy-trinh-tra-hang-hoan-tien",
        "gold_answer": "3-5 ngày làm việc, không tính chủ nhật, ngày lễ và Tết.",
    },
    {
        "id": 3,
        "query": "Sau khi được chấp nhận trả hàng và hoàn tiền người mua phải gửi hàng trong bao lâu?",
        "filter": None,
        "gold_doc": "quy-trinh-tra-hang-hoan-tien",
        "gold_answer": "Trong vòng 6 ngày kể từ khi nhận thông báo gửi trả hàng.",
    },
    {
        "id": 4,
        "query": "Điều kiện bảo hành cơ bản trên Shopee gồm những gì?",
        "filter": None,
        "gold_doc": "chinh-sach-bao-hanh-shopee",
        "gold_answer": "Còn thời hạn bảo hành, còn tem/phiếu bảo hành, lỗi kỹ thuật không do người mua.",
    },
    {
        "id": 5,
        "query": "Hủy đơn do hết hàng hoặc không xác nhận đúng hạn thì bị phí bao nhiêu?",
        "filter": {"audience": "seller"},
        "gold_doc": "quy-dinh-nguoi-ban-shopee-mall",
        "gold_answer": "196.360 VND cho mỗi đơn hàng bị hủy.",
    },
]


def run_benchmark(strategy_name: str, chunker: Any):
    print(f"\n=======================================================")
    print(f"CHẠY BENCHMARK: Chiến lược [{strategy_name}]")
    print(f"=======================================================")

    # 1. Khởi tạo Embedder (ưu tiên LocalEmbedder, fallback sang _mock_embed)
    try:
        embed_fn = LocalEmbedder()
        print("  -> Sử dụng: LocalEmbedder (sentence-transformers)")
    except Exception as e:
        print(f"  -> Lưu ý: Dùng _mock_embed do không tải được model ({e})")
        embed_fn = _mock_embed

    store = EmbeddingStore(embedding_fn=embed_fn)

    # 2. Đọc corpus từ thư mục data/
    data_dir = Path("data/shopee_refund")
    md_files = list(data_dir.glob("*.md"))
    if not md_files:
        md_files = list(data_dir.glob("**/*.md"))

    if not md_files:
        print("LỖI: Không tìm thấy file .md nào trong thư mục data/!")
        return

    all_docs: list[Document] = []
    print(f"  -> Tìm thấy {len(md_files)} file Markdown trong {data_dir}/")

    for file_path in md_files:
        meta, content = parse_markdown_file(file_path)
        doc_stem = file_path.stem

        # Chunk nội dung file
        chunks = chunker.chunk(content)
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc_stem}#{i}",
                content=chunk_text,
                metadata={
                    **meta,
                    "doc_id": doc_stem,
                    "chunk_id": f"{doc_stem}#{i}",
                },
            )
            all_docs.append(chunk_doc)

    store.add_documents(all_docs)
    avg_len = sum(len(d.content) for d in all_docs) / len(all_docs) if all_docs else 0
    print(f"  -> Tổng số Chunks: {len(all_docs)} | Độ dài TB: {avg_len:.1f} ký tự")

    # 3. Chạy 5 câu benchmark query
    print("\n--- KẾT QUẢ TOP-1 TRUY XUẤT CHO 5 BENCHMARK QUERIES ---")
    for q in BENCHMARK_QUERIES:
        q_id = q["id"]
        query_text = q["query"]
        q_filter = q["filter"]

        results = store.search_with_filter(
            query=query_text, top_k=3, metadata_filter=q_filter
        )

        if results:
            top1 = results[0]
            top1_id = top1["id"]
            top1_score = top1.get("score", 0.0)
            preview = top1["content"].replace("\n", " ")[:90] + "..."
            doc_id = top1.get("metadata", {}).get("doc_id", "")
            is_relevant = "Có" if q["gold_doc"] in doc_id else "Cần kiểm tra"

            print(f"\n[Câu {q_id}]: {query_text}")
            print(f"  Filter: {q_filter}")
            print(f"  Top-1 ID: {top1_id} | Score: {top1_score:.4f}")
            print(f"  Trích đoạn: {preview}")
            print(f"  Relevant: {is_relevant}")
        else:
            print(f"\n[Câu {q_id}]: Không tìm thấy kết quả!")

    # 4. A/B TEST BẮT BUỘC CHO CÂU 5 (Có Filter vs Không Filter)
    print("\n-------------------------------------------------------")
    print("A/B TEST CHO CÂU 5: Ảnh hưởng của metadata_filter")
    print("-------------------------------------------------------")
    q5 = BENCHMARK_QUERIES[4]["query"]

    res_with_filter = store.search_with_filter(
        query=q5, top_k=3, metadata_filter={"audience": "seller"}
    )
    res_no_filter = store.search(query=q5, top_k=3)

    print("▶ KHI CÓ FILTER {'audience': 'seller'}:")
    for rank, r in enumerate(res_with_filter, 1):
        print(
            f"   Top-{rank}: {r['id']} (Score: {r['score']:.4f}) [audience: {r.get('metadata', {}).get('audience')}]"
        )

    print("\n▶ KHI KHÔNG CÓ FILTER:")
    for rank, r in enumerate(res_no_filter, 1):
        print(
            f"   Top-{rank}: {r['id']} (Score: {r['score']:.4f}) [audience: {r.get('metadata', {}).get('audience')}]"
        )


if __name__ == "__main__":
    # =====================================================================
    # CHỌN CHIẾN LƯỢC CỦA BẠN (ĐỖ HOÀNG QUÂN: RecursiveChunker)
    # =====================================================================
    my_chunker = RecursiveChunker(chunk_size=500)
    run_benchmark(strategy_name="RecursiveChunker (Đỗ Hoàng Quân)", chunker=my_chunker)