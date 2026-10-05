"""
BÀI TẬP LẬP TRÌNH NLP - BÀI 7
Chủ đề: Xây dựng ứng dụng phân loại cảm xúc đơn giản dùng Word Similarity
Tập từ điển cho trước:
  Positive = ["good", "excellent", "happy", "awesome"]
  Negative = ["bad", "terrible", "sad", "awful"]

Yêu cầu:
  a. Nhập vào một từ tiếng Anh bất kỳ.
  b. Tính độ tương đồng với mỗi từ trong tập Positive và Negative.
  c. Gán nhãn cho từ dựa theo nhóm có similarity cao hơn.
"""

import sys
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets("good")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Tập từ điển phân loại cảm xúc
POSITIVE_WORDS = ["good", "excellent", "happy", "awesome"]
NEGATIVE_WORDS = ["bad", "terrible", "sad", "awful"]


def word_similarity(w1, w2):
    """
    Tính độ tương đồng tốt nhất giữa 2 từ bất kỳ bằng WordNet.
    Thử nghiệm độ đo Path và Wup, đồng thời kiểm tra quan hệ antonyms / synset lemma similarity.
    """
    syns1 = wn.synsets(w1)
    syns2 = wn.synsets(w2)

    if not syns1 or not syns2:
        return 0.0

    best_score = 0.0
    for s1 in syns1:
        for s2 in syns2:
            # Ưu tiên wup similarity
            score = s1.wup_similarity(s2)
            if score is None:
                score = s1.path_similarity(s2)
            if score is not None and score > best_score:
                best_score = score

    return best_score


def classify_word_sentiment(word):
    """
    Tính độ tương đồng của từ đối với tập Positive và Negative, sau đó gán nhãn.
    """
    word = word.strip().lower()
    
    # b. Tính độ tương đồng với mỗi từ trong tập Positive và Negative
    pos_scores = {}
    for pos_w in POSITIVE_WORDS:
        pos_scores[pos_w] = word_similarity(word, pos_w)

    neg_scores = {}
    for neg_w in NEGATIVE_WORDS:
        neg_scores[neg_w] = word_similarity(word, neg_w)

    # Tính điểm tổng hợp (Max similarity hoặc Average similarity)
    avg_pos = sum(pos_scores.values()) / len(pos_scores) if pos_scores else 0.0
    avg_neg = sum(neg_scores.values()) / len(neg_scores) if neg_scores else 0.0

    max_pos = max(pos_scores.values()) if pos_scores else 0.0
    max_neg = max(neg_scores.values()) if neg_scores else 0.0

    # Điểm đánh giá kết hợp: 60% Max + 40% Average
    pos_score_final = 0.6 * max_pos + 0.4 * avg_pos
    neg_score_final = 0.6 * max_neg + 0.4 * avg_neg

    # c. Gán nhãn cho từ dựa theo nhóm có similarity cao hơn
    if pos_score_final > neg_score_final:
        sentiment = "POSITIVE"
    elif neg_score_final > pos_score_final:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL / UNCERTAIN"

    return {
        "word": word,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "avg_pos": avg_pos,
        "avg_neg": avg_neg,
        "max_pos": max_pos,
        "max_neg": max_neg,
        "final_pos": pos_score_final,
        "final_neg": neg_score_final,
        "sentiment": sentiment
    }


def print_classification_result(result):
    print("=" * 75)
    print(f"KẾT QUẢ PHÂN LOẠI CẢM XÚC CHO TỪ: '{result['word'].upper()}'")
    print("=" * 75)

    print("\n* Điểm tương đồng với tập POSITIVE:")
    for w, score in result["pos_scores"].items():
        print(f"   - vs {w:<12}: {score:.4f}")
    print(f"   => Max Positive: {result['max_pos']:.4f} | Trung bình: {result['avg_pos']:.4f} | Điểm Pos: {result['final_pos']:.4f}")

    print("\n* Điểm tương đồng với tập NEGATIVE:")
    for w, score in result["neg_scores"].items():
        print(f"   - vs {w:<12}: {score:.4f}")
    print(f"   => Max Negative: {result['max_neg']:.4f} | Trung bình: {result['avg_neg']:.4f} | Điểm Neg: {result['final_neg']:.4f}")

    print("\n" + "-" * 75)
    print(f"==> KẾT QUẢ GÁN NHÃN: [{result['sentiment']}]")
    print("-" * 75)


def main():
    print("=" * 80)
    print("BÀI 7: ỨNG DỤNG PHÂN LOẠI CẢM XÚC DÙNG WORD SIMILARITY")
    print(f"Tập Positive: {POSITIVE_WORDS}")
    print(f"Tập Negative: {NEGATIVE_WORDS}")
    print("=" * 80)

    # Chạy thử nghiệm tự động trên tập từ mẫu
    demo_words = ["joyful", "horrible", "delightful", "tragic", "pleasant", "nasty"]
    print("\n--- [DEMO] Chạy kiểm thử tự động với danh sách từ mẫu ---")
    for dw in demo_words:
        res = classify_word_sentiment(dw)
        print(f"Từ: '{dw:<12}' -> Nhãn: {res['sentiment']:<10} (Pos: {res['final_pos']:.3f} vs Neg: {res['final_neg']:.3f})")

    # a. Cho phép người dùng nhập từ tiếng Anh bất kỳ
    print("\n" + "=" * 80)
    if sys.stdin.isatty():
        user_input = input("Nhập vào một từ tiếng Anh bất kỳ (hoặc nhấn Enter để chọn 'wonderful'): ").strip()
    else:
        user_input = ""

    if not user_input:
        user_input = "wonderful"
        print(f"[*] Chế độ tự động: Đang kiểm tra từ mẫu '{user_input}'...")

    result = classify_word_sentiment(user_input)
    print_classification_result(result)


if __name__ == "__main__":
    main()
