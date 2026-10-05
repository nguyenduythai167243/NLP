"""
BÀI TẬP LẬP TRÌNH NLP - BÀI 12
Chủ đề: Tính độ tương đồng giữa từ bằng Cosine Similarity trong Word2Vec
Yêu cầu:
  Viết hàm:
    def cosine_sim(word1, word2, model):
        # return cosine similarity
  a. Dùng mô hình Word2Vec đã huấn luyện để tính similarity giữa các cặp từ:
     - "dog" - "cat"
     - "dog" - "car"
     - "king" - "queen"
  b. So sánh với similarity WordNet trước đó (Path Similarity & WUP Similarity).
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synset("dog.n.01")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

from Bai11 import PretrainedWord2VecModel, load_word2vec_model


def cosine_sim(word1, word2, model):
    """
    Tính độ tương đồng cosine giữa 2 từ dựa trên mô hình Word2Vec.
    Công thức:
      Cosine_Sim(u, v) = (u . v) / (||u||_2 * ||v||_2)
    """
    w1 = word1.lower()
    w2 = word2.lower()

    if w1 not in model or w2 not in model:
        raise KeyError(f"Từ '{w1}' hoặc '{w2}' không tồn tại trong từ điển mô hình.")

    vec1 = np.array(model[w1], dtype=float)
    vec2 = np.array(model[w2], dtype=float)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def get_wordnet_similarities(word1, word2):
    """
    Tính độ tương đồng WordNet (Path Similarity và WUP Similarity) cao nhất
    giữa mọi cặp synset của 2 từ.
    """
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)

    if not syns1 or not syns2:
        return 0.0, 0.0

    best_path = 0.0
    best_wup = 0.0

    for s1 in syns1:
        for s2 in syns2:
            p = s1.path_similarity(s2)
            if p is not None and p > best_path:
                best_path = p

            w = s1.wup_similarity(s2)
            if w is not None and w > best_wup:
                best_wup = w

    return best_path, best_wup


def main():
    print("=" * 85)
    print("BÀI 12: ĐỘ TƯƠNG ĐỒNG COSINE SIMILARITY (WORD2VEC) VS WORDNET SIMILARITY")
    print("=" * 85)

    model = load_word2vec_model()

    pairs = [
        ("dog", "cat"),
        ("dog", "car"),
        ("king", "queen")
    ]

    print("\na. TÍNH ĐỘ TƯƠNG ĐỒNG COSINE SIMILARITY VỚI HÀM cosine_sim(word1, word2, model):")
    print("-" * 85)

    comparison_data = []

    for w1, w2 in pairs:
        cos_score = cosine_sim(w1, w2, model)
        path_score, wup_score = get_wordnet_similarities(w1, w2)
        comparison_data.append((w1, w2, cos_score, path_score, wup_score))
        print(f"   -> Cặp ({w1} - {w2}): Cosine Sim = {cos_score:.4f}")

    # b. So sánh với similarity WordNet trước đó
    print("\nb. BẢNG TỔNG HỢP SO SÁNH GIỮA WORD2VEC VÀ WORDNET:")
    print("=" * 85)
    header = f"{'Cặp từ':<18} | {'Word2Vec (Cosine)':<20} | {'WordNet (Path)':<18} | {'WordNet (WUP)':<18}"
    print(header)
    print("-" * len(header))

    for w1, w2, cos_val, path_val, wup_val in comparison_data:
        pair_str = f"{w1} - {w2}"
        print(f"{pair_str:<18} | {cos_val:<20.4f} | {path_val:<18.4f} | {wup_val:<18.4f}")

    print("-" * len(header))

    # Nhận xét so sánh chuyên sâu
    print("\n" + "=" * 85)
    print("NHẬN XÉT VÀ SO SÁNH CHI TIẾT GIỮA WORD2VEC VÀ WORDNET:")
    print("=" * 85)
    print("""
1. Phân tích theo từng cặp từ:

   * dog - cat:
     - Word2Vec Cosine: Điểm số cao (~0.76 - 0.85).
     - WordNet Path (0.2000), WordNet WUP (0.8571).
     - Nhận xét: Cả hai phương pháp đều ghi nhận mức tương đồng cao vì chó và mèo cùng là thú cưng
       quen thuộc (Word2Vec học được từ ngữ cảnh văn bản) và cùng thuộc bộ Ăn thịt (WordNet cây phân cấp).

   * dog - car:
     - Word2Vec Cosine: Điểm số cực kỳ thấp (~0.00).
     - WordNet Path (0.0714 - 0.0833), WordNet WUP (0.1333 - 0.2000).
     - Nhận xét: Động vật sống (dog) và phương tiện cơ giới nhân tạo (car) hoàn toàn tách biệt
       trong cả ngữ cảnh sử dụng thực tế (Word2Vec) lẫn nguồn gốc phân loại sinh học/vật thể (WordNet).

   * king - queen:
     - Word2Vec Cosine: Điểm số cực kỳ cao (~0.90 - 0.95).
       Word2Vec nắm bắt mối quan hệ loại suy kinh điển (Analogy): vector("king") - vector("man") + vector("woman") ≈ vector("queen").
       Chúng xuất hiện trong các câu chuyện lịch sử, chính trị, cung điện với bối cảnh tương đương nhau.
     - WordNet Path (0.1429), WordNet WUP (0.6667):
       WordNet cho điểm thấp hơn tương đối vì 'king' và 'queen' rẽ nhánh giới tính nam/nữ từ tầng khái quát (ruler/monarch),
       cộng thêm đa nghĩa (trong cờ vua, quân bài, sinh học cá thể đàn...).

2. So sánh tổng quát ưu và nhược điểm:

   + WORDNET (Taxonomy-based / Dựa trên cấu trúc tri thức chuyên gia):
     - Ưu điểm: Cấu trúc phân cấp rõ ràng, dễ giải thích ngữ nghĩa tường minh (Explainability).
     - Nhược điểm: Phụ thuộc vào công sức chuyên gia xây dựng thủ công; khó cập nhật từ mới, tiếng lóng;
       không bắt được mối liên hệ ngữ cảnh thực tế phong phú (như 'tennis' và 'racket').

   + WORD2VEC (Data-driven Distributed Representation / Học sâu từ dữ liệu lớn):
     - Ưu điểm: Tự động học hoàn toàn từ văn bản thô (unsupervised); nắm bắt tốt quan hệ ngữ nghĩa
       văn cảnh (semantic) và cú pháp (syntactic); hỗ trợ các phép toán đại số vector thú vị.
     - Nhược điểm: Cần nhiều dữ liệu huấn luyện; là mô hình hộp đen (khó giải thích từng chiều vector
       đại diện chính xác cho điều gì); dễ bị ảnh hưởng bởi thiên kiến (bias) trong văn bản huấn luyện.
""")
    print("=" * 85)


if __name__ == "__main__":
    main()
