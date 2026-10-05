
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synset("car.n.01")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def calculate_best_path_similarity(word1, word2):
    """
    Tính path similarity cao nhất giữa mọi cặp synset (danh từ) của 2 từ.
    Trả về (best_score, best_syn1, best_syn2).
    """
    synsets1 = wn.synsets(word1, pos=wn.NOUN)
    synsets2 = wn.synsets(word2, pos=wn.NOUN)

    best_score = -1.0
    best_pair = (None, None)

    for s1 in synsets1:
        for s2 in synsets2:
            score = s1.path_similarity(s2)
            if score is not None and score > best_score:
                best_score = score
                best_pair = (s1, s2)

    return best_score, best_pair[0], best_pair[1]


def main():
    word_pairs = [
        ("car", "automobile"),
        ("dog", "cat"),
        ("dog", "wolf"),
        ("tree", "car")
    ]

    print("=" * 80)
    print("BÀI 4: TÍNH ĐỘ TƯƠNG ĐỒNG BẰNG PATH SIMILARITY TRONG WORDNET")
    print("Công thức: Path_Similarity = 1 / (shortest_path_distance + 1)")
    print("Phạm vi giá trị: (0, 1] - Càng gần 1 thì ngữ nghĩa càng tương đồng")
    print("=" * 80)

    results = []

    print(f"\n{'Cặp từ':<25} | {'Synset 1':<15} | {'Synset 2':<15} | {'Path Similarity':<15}")
    print("-" * 78)

    for w1, w2 in word_pairs:
        score, s1, s2 = calculate_best_path_similarity(w1, w2)
        s1_name = s1.name() if s1 else "N/A"
        s2_name = s2.name() if s2 else "N/A"
        score_str = f"{score:.4f}" if score is not None else "None"
        pair_str = f"{w1} - {w2}"
        print(f"{pair_str:<25} | {s1_name:<15} | {s2_name:<15} | {score_str:<15}")
        results.append((w1, w2, score, s1, s2))

    print("-" * 78)

    # Nhận xét chi tiết
    print("\n" + "=" * 80)
    print("NHẬN XÉT VỀ CÁC GIÁ TRỊ SIMILARITY:")
    print("=" * 80)

    print("""
1. car - automobile (Similarity = 1.0000):
   -> Điểm số tuyệt đối tối đa (1.0).
   -> Lý do: 'car' và 'automobile' là từ đồng nghĩa hoàn toàn (synonyms), cùng chung
      synset ('car.n.01'). Khoảng cách đường đi (path distance) bằng 0 => 1/(0+1) = 1.0.

2. dog - wolf (Similarity ≈ 0.3333 hoặc 0.5 tùy synset):
   -> Điểm tương đồng rất cao.
   -> Lý do: Cả chó (dog) và chó sói (wolf) đều thuộc họ Chó (Canidae). Tổ tiên chung
      gần nhất (LCS) nằm ở cấp rất gần trong cây phân loại IS-A (canine.n.02),
      khoảng cách đường đi ngắn.

3. dog - cat (Similarity ≈ 0.2000):
   -> Điểm tương đồng mức trung bình.
   -> Lý do: Chó và mèo đều là động vật có vú, ăn thịt (Carnivore) và là thú nuôi quen thuộc.
      Tuy nhiên chúng rẽ nhánh sang hai họ khác nhau (Canidae vs Felidae),
      do đó khoảng cách đường đi trên cây WordNet xa hơn so với cặp dog - wolf.

4. tree - car (Similarity ≈ 0.0714 - 0.0833):
   -> Điểm tương đồng cực kỳ thấp (tiệm cận 0).
   -> Lý do: 'tree' thuộc nhánh thực vật / sinh vật sống (living thing / plant), còn 'car'
      thuộc nhánh đồ vật nhân tạo / phương tiện (artifact / vehicle). Hai nhánh này tách rời
      nhau từ đỉnh cây WordNet (nút chung entity.n.01), khoảng cách đường đi rất dài,
      chứng tỏ hai khái niệm hầu như không có liên hệ ngữ nghĩa trong phân loại tự nhiên.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
