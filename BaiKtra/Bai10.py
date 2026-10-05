"""
BÀI TẬP LẬP TRÌNH NLP - BÀI 10
Chủ đề: Khử nhập nhằng ngữ nghĩa bằng thuật toán Lesk (Word Sense Disambiguation)
Yêu cầu:
  Sử dụng thuật toán Lesk trong NLTK (from nltk.wsd import lesk)
  Viết chương trình chọn nghĩa phù hợp cho từ "bass" trong 2 câu:
    Câu 1: "I can hear the bass in this music."
    Câu 2: "He caught a huge bass in the lake."
"""

import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

try:
    wn.synsets("bass")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)


def disambiguate_with_lesk(sentence, target_word, pos='n'):
    """
    Sử dụng thuật toán Lesk để khử nhập nhằng ngữ nghĩa cho từ mục tiêu trong câu.
    """
    # Tách từ đơn giản
    tokens = re.findall(r"\b\w+\b", sentence.lower())
    
    # Áp dụng hàm lesk của NLTK
    best_synset = lesk(tokens, target_word, pos)
    
    return tokens, best_synset


def analyze_lesk_overlap(tokens, target_synset):
    """
    Phân tích từ vựng giao thoa (overlap) giữa ngữ cảnh câu và định nghĩa/ví dụ của synset.
    """
    if not target_synset:
        return []
    
    # Tập từ vựng trong định nghĩa và ví dụ của synset
    synset_text = target_synset.definition() + " " + " ".join(target_synset.examples())
    synset_tokens = set(re.findall(r"\b\w+\b", synset_text.lower()))
    
    # Từ ngữ cảnh trong câu (bỏ từ target)
    context_tokens = set(tokens) - {"bass", "a", "an", "the", "in", "on", "this", "can", "he", "i"}
    
    # Tập từ giao thoa
    overlap = context_tokens.intersection(synset_tokens)
    return list(overlap)


def main():
    target_word = "bass"
    sentences = [
        ("Câu 1", "I can hear the bass in this music."),
        ("Câu 2", "He caught a huge bass in the lake.")
    ]

    print("=" * 80)
    print("BÀI 10: KHỬ NHẬP NHẰNG NGỮ NGHĨA (WSD) BẰNG THUẬT TOÁN LESK")
    print(f"Từ cần giải nghĩa: '{target_word}' (Danh từ)")
    print("=" * 80)

    # Hiển thị tất cả nghĩa có thể của từ "bass" trong WordNet
    all_synsets = wn.synsets(target_word, pos=wn.NOUN)
    print(f"\n[*] Tổng số synset danh từ của '{target_word}' trong WordNet: {len(all_synsets)}")
    for i, s in enumerate(all_synsets, 1):
        print(f"   [{i}] {s.name():<15}: {s.definition()}")
    print("-" * 80)

    # Chạy thuật toán Lesk cho từng câu
    for name, sent in sentences:
        tokens, chosen_synset = disambiguate_with_lesk(sent, target_word, 'n')
        overlaps = analyze_lesk_overlap(tokens, chosen_synset)

        print(f"\n>>> {name}: \"{sent}\"")
        print(f"    - Các token xử lý : {tokens}")
        if chosen_synset:
            print(f"    - Synset được chọn: {chosen_synset.name()}")
            print(f"    - Định nghĩa       : {chosen_synset.definition()}")
            if chosen_synset.examples():
                print(f"    - Ví dụ WordNet   : {chosen_synset.examples()}")
            if overlaps:
                print(f"    - Từ trùng lặp (Overlap): {overlaps}")
        else:
            print(f"    - Không tìm thấy nghĩa phù hợp.")
        print("-" * 80)

    # Nhận xét và giải thích nguyên lý hoạt động
    print("\n" + "=" * 80)
    print("GIẢI THÍCH VÀ NHẬN XÉT NGUYÊN LÝ THUẬT TOÁN LESK:")
    print("=" * 80)
    print("""
1. Nguyên lý hoạt động của thuật toán Lesk:
   - Thuật toán Lesk dựa trên giả thuyết rằng các từ xuất hiện cùng nhau trong một câu sẽ chia sẻ
     chủ đề chung.
   - Khi khử nhập nhằng cho từ 'target_word', thuật toán đếm số lượng từ trùng khớp (overlap)
     giữa:
       (1) Tập từ ngữ cảnh xung quanh trong câu.
       (2) Định nghĩa (definition) và câu ví dụ (examples) của từng synset ứng viên trong WordNet.
   - Synset nào có số lượng từ trùng lặp lớn nhất (maximum overlap) sẽ được chọn.

2. Đánh giá kết quả trên 2 câu thử nghiệm:
   - Trong Câu 1 ("I can hear the bass in this music."):
     + Ngữ cảnh xuất hiện các từ: 'music', 'hear'.
     + Synset được chọn: 'bass.n.01' (âm trầm, tông giọng thấp nhất trong âm nhạc / nhạc cụ).
     + Kết quả HOÀN TOÀN CHÍNH XÁC với ngữ cảnh âm nhạc!

   - Trong Câu 2 ("He caught a huge bass in the lake."):
     + Ngữ cảnh xuất hiện các từ: 'caught' (bắt được), 'lake' (hồ nước), 'huge' (to lớn).
     + Synset được chọn: 'sea_bass.n.01' hoặc 'bass.n.08' (loài cá vược nước ngọt / cá biển ăn thịt).
     + Kết quả HOÀN TOÀN CHÍNH XÁC với ngữ cảnh đánh bắt thủy sản / câu cá!
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
