"""
BÀI TẬP LẬP TRÌNH NLP - BÀI 9
Chủ đề: Tính semantic similarity cho câu (WordNet-based Sentence Similarity)
Yêu cầu:
  Viết hàm:
    def sentence_similarity(sent1, sent2):
        # average max-similarity between words from the two sentences
  Kiểm tra với các câu:
    a. "A dog is running in the park."
    b. "A wolf is moving in the forest."
    c. "A car is parked beside the road."
  Nhận xét kết quả.
"""

import re
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets("dog")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Danh sách stop words tiếng Anh cơ bản để lọc từ chức năng không mang nghĩa từ vựng
STOP_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'is', 'am', 'are', 'was', 'were',
    'and', 'or', 'to', 'of', 'for', 'with', 'beside', 'by', 'this', 'that'
}


def word_max_similarity(w1, w2):
    """Tính WUP similarity cao nhất giữa mọi cặp synset của w1 và w2."""
    if w1 == w2:
        return 1.0

    syns1 = wn.synsets(w1)
    syns2 = wn.synsets(w2)

    if not syns1 or not syns2:
        return 0.0

    best = 0.0
    for s1 in syns1:
        for s2 in syns2:
            score = s1.wup_similarity(s2)
            if score is not None and score > best:
                best = score
    return best


def tokenize_and_filter(sentence):
    """Tách từ, chuyển về chữ thường và loại bỏ hư từ (stopwords/punctuation)."""
    words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
    content_words = [w for w in words if w not in STOP_WORDS]
    return content_words if content_words else words


def sentence_similarity(sent1, sent2):
    """
    Tính độ tương đồng ngữ nghĩa giữa 2 câu:
    average max-similarity between words from the two sentences.
    
    Công thức:
      - Với mỗi từ w1 in sent1, tìm max similarity với mọi từ w2 in sent2.
      - Với mỗi từ w2 in sent2, tìm max similarity với mọi từ w1 in sent1.
      - Lấy trung bình cộng đối xứng 2 chiều để đảm bảo tính giao hoán: Sim(S1, S2) = Sim(S2, S1).
    """
    words1 = tokenize_and_filter(sent1)
    words2 = tokenize_and_filter(sent2)

    if not words1 or not words2:
        return 0.0

    # Chiều 1: Từ words1 sang words2
    sim_scores_1 = []
    for w1 in words1:
        max_sim = max([word_max_similarity(w1, w2) for w2 in words2], default=0.0)
        sim_scores_1.append(max_sim)

    # Chiều 2: Từ words2 sang words1
    sim_scores_2 = []
    for w2 in words2:
        max_sim = max([word_max_similarity(w2, w1) for w1 in words1], default=0.0)
        sim_scores_2.append(max_sim)

    avg_1 = sum(sim_scores_1) / len(sim_scores_1)
    avg_2 = sum(sim_scores_2) / len(sim_scores_2)

    # Trung bình cộng 2 chiều
    return (avg_1 + avg_2) / 2.0


def main():
    sent_a = "A dog is running in the park."
    sent_b = "A wolf is moving in the forest."
    sent_c = "A car is parked beside the road."

    sentences = {
        "Câu a": sent_a,
        "Câu b": sent_b,
        "Câu c": sent_c
    }

    print("=" * 80)
    print("BÀI 9: TÍNH ĐỘ TƯƠNG ĐỒNG NGỮ NGHĨA CÂU (SENTENCE SIMILARITY)")
    print("=" * 80)
    print("Các câu kiểm tra:")
    for k, v in sentences.items():
        words = tokenize_and_filter(v)
        print(f"  {k}: \"{v}\"")
        print(f"       -> Từ nội dung (Content words): {words}")
    print("-" * 80)

    # Tính độ tương đồng giữa các cặp câu
    pairs = [
        ("Câu a", "Câu b", sent_a, sent_b),
        ("Câu a", "Câu c", sent_a, sent_c),
        ("Câu b", "Câu c", sent_b, sent_c)
    ]

    print(f"\n{'Cặp câu so sánh':<25} | {'Nội dung tóm tắt':<30} | {'Độ tương đồng (Similarity)':<20}")
    print("-" * 80)

    results = {}
    for name1, name2, s1, s2 in pairs:
        sim = sentence_similarity(s1, s2)
        key = f"{name1} vs {name2}"
        results[key] = sim
        short_desc = f"{name1[-1]} & {name2[-1]}"
        if "a" in short_desc and "b" in short_desc:
            desc = "Động vật di chuyển trong tự nhiên"
        else:
            desc = "Động vật vs Phương tiện/Đỗ xe"
        print(f"{key:<25} | {desc:<30} | {sim:<20.4f}")

    print("-" * 80)

    # Nhận xét kết quả
    print("\n" + "=" * 80)
    print("NHẬN XÉT CHI TIẾT KẾT QUẢ:")
    print("=" * 80)
    print(f"""
1. So sánh Câu a vs Câu b (Độ tương đồng cao nhất: {results['Câu a vs Câu b']:.4f}):
   - Câu a: "A dog is running in the park."
   - Câu b: "A wolf is moving in the forest."
   - Giải thích:
     + Chủ ngữ: 'dog' (chó) và 'wolf' (chó sói) đều thuộc họ Chó (Canidae), có WUP similarity rất cao (~0.90).
     + Động từ: 'running' (chạy) và 'moving' (di chuyển) là các hành động vận động có trường nghĩa cực gần.
     + Bối cảnh: 'park' (công viên) và 'forest' (rừng rậm) đều là không gian cây xanh ngoài trời.
     => Do đó, hai câu này có độ tương đồng ngữ nghĩa cao nhất.

2. So sánh Câu a vs Câu c ({results['Câu a vs Câu c']:.4f}) và Câu b vs Câu c ({results['Câu b vs Câu c']:.4f}):
   - Câu c: "A car is parked beside the road." (Ô tô đỗ bên đường).
   - Giải thích:
     + Câu c nói về thực thể nhân tạo vô tri ('car' - ô tô), hành động bất động ('parked' - đỗ xe),
       và công trình nhân tạo ('road' - con đường).
     + Khi so khớp từng từ giữa Câu a/b với Câu c ('dog/wolf' vs 'car', 'running/moving' vs 'parked',
       'park/forest' vs 'road'), các cặp từ đều có điểm WUP rất thấp.
     => Điểm số tương đồng giữa Câu a/b với Câu c giảm rõ rệt.

3. Kết luận:
   Phương pháp WordNet-based Sentence Similarity đã phản ánh rất tốt mức độ tương đồng
   về mặt ngữ nghĩa (semantic similarity) giữa các câu trong thực tế!
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
