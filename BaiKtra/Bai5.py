

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synset("car.n.01")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def get_best_similarities(word1, word2):
    """
    Tìm cặp synset danh từ có độ đo tương đồng tốt nhất,
    trả về cả path_similarity và wup_similarity tương ứng.
    """
    syns1 = wn.synsets(word1, pos=wn.NOUN)
    syns2 = wn.synsets(word2, pos=wn.NOUN)

    best_wup = -1.0
    best_path = -1.0
    best_wup_pair = (None, None)
    best_path_pair = (None, None)

    # Tìm max wup
    for s1 in syns1:
        for s2 in syns2:
            w = s1.wup_similarity(s2)
            if w is not None and w > best_wup:
                best_wup = w
                best_wup_pair = (s1, s2)

            p = s1.path_similarity(s2)
            if p is not None and p > best_path:
                best_path = p
                best_path_pair = (s1, s2)

    return best_path, best_wup, best_path_pair, best_wup_pair


def main():
    word_pairs = [
        ("car", "automobile"),
        ("dog", "wolf"),
        ("dog", "cat"),
        ("tree", "car")
    ]

    print("=" * 85)
    print("BÀI 5: SO SÁNH ĐỘ TƯƠNG ĐỒNG PATH SIMILARITY VÀ WU-PALMER (WUP) SIMILARITY")
    print("=" * 85)
    print("1. Công thức:")
    print("   - Path Similarity = 1 / (shortest_path_distance + 1)")
    print("   - Wu-Palmer Sim   = 2 * depth(LCS) / (depth(synset1) + depth(synset2))")
    print("     (Trong đó LCS là Least Common Subsumer - tổ tiên chung gần nhất)")
    print("-" * 85)

    print(f"\n{'Cặp từ':<20} | {'Path Similarity':<18} | {'Wu-Palmer (WUP)':<18} | {'Chênh lệch (Δ)':<15}")
    print("-" * 85)

    for w1, w2 in word_pairs:
        path_sim, wup_sim, _, _ = get_best_similarities(w1, w2)
        diff = (wup_sim - path_sim) if (wup_sim is not None and path_sim is not None) else 0.0
        pair_name = f"{w1} - {w2}"
        print(f"{pair_name:<20} | {path_sim:<18.4f} | {wup_sim:<18.4f} | {diff:+<15.4f}")

    print("-" * 85)

    print("\n" + "=" * 85)
    print("SO SÁNH VÀ NHẬN XÉT CHI TIẾT GIỮA HAI ĐỘ ĐO:")
    print("=" * 85)
    print("""
1. Giá trị của Wu-Palmer luôn lớn hơn hoặc bằng Path Similarity:
   - Ví dụ:
     + dog - wolf: Path = 0.3333 -> WUP = 0.9000 (tăng đáng kể)
     + dog - cat : Path = 0.2000 -> WUP = 0.8571 (tăng đáng kể)
     + tree - car: Path = 0.0714 -> WUP = 0.1333

2. Lý giải nguyên nhân sự khác biệt:
   - Path Similarity:
     + Chỉ phụ thuộc đơn thuần vào số bước nhảy (độ dài đường đi) giữa 2 khái niệm.
     + Hạn chế: Coi mọi bước nhảy ở các tầng khác nhau là bình đẳng (khoảng cách 2 bước ở
       tầng sâu cụ thể bị coi ngang bằng khoảng cách 2 bước ở tầng khái quát cao).
   - Wu-Palmer Similarity:
     + Tích hợp độ sâu (depth) của nút tổ tiên chung gần nhất (LCS).
     + Nếu hai từ nằm rất sâu trong cây phân loại và gặp nhau ở một nút tổ tiên cũng rất sâu
       (ví dụ: 'canine' - độ sâu lớn), WUP sẽ phản ánh chính xác rằng hai loài này cực kỳ
       gần gũi và tương đồng cao.

3. Trường hợp đồng nghĩa tuyệt đối (car - automobile):
   - Cả 2 độ đo đều đạt giá trị cực đại = 1.0000, do khoảng cách bằng 0 và depth(LCS) = depth(car).

4. Kết luận ứng dụng:
   - Wu-Palmer phản ánh trực giác ngữ nghĩa con người tự nhiên hơn và có độ phân giải tốt hơn
     khi so sánh các từ thuộc tầng phân cấp sâu trong phân loại khoa học và đời sống.
""")
    print("=" * 85)


if __name__ == "__main__":
    main()
