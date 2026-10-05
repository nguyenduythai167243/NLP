
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

try:
    wn.synsets("crane")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)


def extract_word(sentence, target):
    """Trích xuất từ mục tiêu ra khỏi câu bằng regex/tokenization."""
    pattern = rf"\b({re.escape(target)})\b"
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def main():
    sentence = "The crane is standing near the river."
    target = "crane"

    print("=" * 80)
    print("BÀI 3: TÌM NGHĨA GẦN ĐÚNG CỦA TỪ TRONG CÂU")
    print(f"Câu gốc: \"{sentence}\"")
    print("=" * 80)

    # a. Trích xuất từ "crane"
    extracted_word = extract_word(sentence, target)
    print(f"\na. Trích xuất từ khóa:")
    print(f"   -> Kết quả trích xuất: '{extracted_word}' từ câu đã cho.")

    # b. Dùng WordNet liệt kê tất cả nghĩa có thể
    synsets = wn.synsets(extracted_word)
    print(f"\nb. Liệt kê tất cả các nghĩa có thể của từ '{extracted_word}' ({len(synsets)} nghĩa):")
    print("-" * 80)

    for i, syn in enumerate(synsets, 1):
        pos_str = "Danh từ" if syn.pos() == 'n' else ("Động từ" if syn.pos() == 'v' else syn.pos())
        print(f"[{i}] Synset: {syn.name()} ({pos_str})")
        print(f"    Định nghĩa: {syn.definition()}")
        if syn.examples():
            print(f"    Ví dụ     : {syn.examples()}")

    # c. Dự đoán nghĩa phù hợp nhất
    print("\nc. Dự đoán nghĩa phù hợp nhất dựa vào ngữ cảnh câu:")
    print("-" * 80)

    # Phân tích ngữ cảnh từ câu:
    # "standing near the river" -> sông (river), đứng (standing)
    # So sánh 2 nhóm nghĩa chính:
    # - Animal (loài chim sếu lội nước, sống gần bờ sông/đầm lầy): crane.n.05
    # - Machine (máy cẩu nâng hạ hàng hóa dùng trong xây dựng): crane.n.02
    print("1. Phân tích các từ khóa ngữ cảnh trong câu:")
    tokens = [w.lower() for w in re.findall(r"\b\w+\b", sentence)]
    context_keywords = [w for w in tokens if w not in {"the", "is", "crane"}]
    print(f"   - Từ ngữ cảnh (Context words): {context_keywords}")
    print("   - 'river' (dòng sông): Môi trường sống tự nhiên của chim lội nước (wading bird).")
    print("   - 'standing' (đang đứng): Hành vi sinh học của sinh vật sống.")

    # Áp dụng thuật toán Lesk để kiểm tra khách quan
    tokens_sentence = sentence.replace(".", "").split()
    best_lesk = lesk(tokens_sentence, extracted_word, 'n')
    print(f"\n2. Kết quả từ thuật toán Word Sense Disambiguation (Lesk):")
    if best_lesk:
        print(f"   - Synset gợi ý: {best_lesk.name()}")
        print(f"   - Định nghĩa  : {best_lesk.definition()}")

    # Đánh giá 2 nghĩa chính:
    animal_syn = wn.synset("crane.n.05")
    machine_syn = wn.synset("crane.n.02")

    print("\n3. So sánh đối chiếu 2 trường nghĩa tiêu biểu (Animal vs Machine):")
    print(f"   [Animal]  {animal_syn.name()} : {animal_syn.definition()}")
    print(f"   [Machine] {machine_syn.name()} : {machine_syn.definition()}")

    print("\n=> KẾT LUẬN DỰ ĐOÁN:")
    print("   Nghĩa phù hợp nhất là: ANIMAL (Loài chim sếu)")
    print(f"   Tên synset: {animal_syn.name()}")
    print(f"   Định nghĩa: \"{animal_syn.definition()}\"")
    print("   Lý do: Trong câu, từ 'crane' kết hợp với cụm 'standing near the river' (đang đứng gần bờ sông).")
    print("   Sông (river) là môi trường sống đặc trưng của loài chim sếu lội nước (wading bird).")
    print("   Cần cẩu/máy cẩu công nghiệp (machine) thông thường đặt ở công trường, bến cảng và không mang đặc tính đứng tự nhiên ven sông.")
    print("=" * 80)


if __name__ == "__main__":
    main()
