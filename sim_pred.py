from src.chunking import compute_similarity
from src.embeddings import LocalEmbedder

pairs = [
    ('Làm thế nào để đổi trả hàng trên Shopee?', 'Quy trình hoàn tiền và trả hàng cho người mua.'),
    ('Đơn hàng được giao đúng hạn.', 'Đơn hàng bị giao trễ hạn.'),
    ('Quy định trả hàng hoàn tiền Shopee Mall.', 'Cài đặt driver card màn hình NVIDIA trên Ubuntu.'),
    ('Người mua được hoàn bao nhiêu tiền?', 'Người bán Shopee Mall bị phạt phí bao nhiêu?'),
    ('Shopee Mall cam kết hàng chính hãng 100%.', 'Hôm nay trời nắng đẹp tôi đi chơi ở trung tâm thương mại Mall.')
]

embedder = LocalEmbedder()
for i, (a, b) in enumerate(pairs, 1):
    sim = compute_similarity(embedder(a), embedder(b))
    print(f'Cặp {i}: {sim:.4f}')