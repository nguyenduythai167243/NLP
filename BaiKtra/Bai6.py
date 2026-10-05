
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synset("ship.n.01")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def wordnet_similarity(word1, word2):
    """
    Tính độ tương đồng WordNet cao nhất giữa mọi cặp synset của 2 từ theo chuẩn đề bài.
    
    Các bước:
      a. Lấy tất cả synset cho từng từ.
      b. Tính WUP similarity cho mọi cặp synset.
      c. Trả về giá trị similarity cao nhất.
    """
    # a. Lấy tất cả synset cho từng từ
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    # Nếu một trong hai từ không có trong WordNet
    if not synsets1 or not synsets2:
        return 0.0

    max_similarity = 0.0

    # b. Tính WUP similarity cho mọi cặp synset
    for s1 in synsets1:
        for s2 in synsets2:
            score = s1.wup_similarity(s2)
            if score is not None and score > max_similarity:
                max_similarity = score

    # c. Trả về giá trị similarity cao nhất
    return max_similarity


def wordnet_similarity_detailed(word1, word2):
    """
    Phiên bản mở rộng trả về thêm thông tin cặp synset đạt điểm cao nhất để phân tích.
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return 0.0, None, None

    max_similarity = 0.0
    best_pair = (None, None)

    for s1 in synsets1:
        for s2 in synsets2:
            score = s1.wup_similarity(s2)
            if score is not None and score > max_similarity:
                max_similarity = score
                best_pair = (s1, s2)

    return max_similarity, best_pair[0], best_pair[1]


def main():
    test_pairs = [
        ("ship", "boat"),
        ("good", "bad"),
        ("food", "fruit"),
        ("computer", "laptop")
    ]

    print("=" * 80)
    print("BÀI 6: HÀM TÍNH ĐỘ TƯƠNG ĐỒNG TỪ VỰNG DỰA TRÊN WORDNET (WUP)")
    print("=" * 80)

    print(f"\n{'Cặp từ':<22} | {'Max WUP Sim':<15} | {'Best Synset 1':<18} | {'Best Synset 2':<18}")
    print("-" * 80)

    for w1, w2 in test_pairs:
        # Gọi hàm chính theo đúng yêu cầu đề bài
        sim_val = wordnet_similarity(w1, w2)
        
        # Lấy thêm chi tiết synset
        _, best_s1, best_s2 = wordnet_similarity_detailed(w1, w2)
        s1_str = best_s1.name() if best_s1 else "N/A"
        s2_str = best_s2.name() if best_s2 else "N/A"

        pair_name = f"{w1} - {w2}"
        print(f"{pair_name:<22} | {sim_val:<15.4f} | {s1_str:<18} | {s2_str:<18}")

    print("-" * 80)

    print("\n" + "=" * 80)
    print("PHÂN TÍCH KẾT QUẢ TỪNG CẶP:")
    print("=" * 80)
    print("""
1. ship - boat (WUP = 0.9091):
   -> Cực kỳ cao vì cùng là phương tiện giao thông đường thủy (vessel / watercraft).
   -> Khoảng cách phân cấp rất gần trên phả hệ WordNet.

2. good - bad (WUP = 0.6667 hoặc thấp hơn tùy POS):
   -> Lưu ý quan trọng: 'good' và 'bad' thường là tính từ (Adjective) và là từ trái nghĩa (Antonyms).
   -> Trong WordNet chuẩn, độ đo WUP được định nghĩa trên cây phân cấp danh từ / động từ.
      Các synset danh từ của good/bad (như sự tốt đẹp vs điều xấu) có thể tìm được tổ tiên chung
      ở tầng trừu tượng (moral condition, quality), do đó vẫn cho ra giá trị tương đối cao.

3. food - fruit (WUP = 0.8889):
   -> Rất cao vì 'fruit' là hyponym trực tiếp hoặc gián tiếp của 'food' (trái cây là đồ ăn).
   -> Có quan hệ IS-A chặt chẽ.

4. computer - laptop (WUP = 0.8889 hoặc cao hơn):
   -> Rất cao vì 'laptop' (máy tính xách tay) là một loại con (hyponym) trực tiếp của 'computer'
      (hoặc portable computer thuộc computer).
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
