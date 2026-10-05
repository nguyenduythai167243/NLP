

import nltk
from nltk.corpus import wordnet as wn

# Tải tài nguyên WordNet nếu chưa có
try:
    wn.synsets('bank')
except LookupError:
    print("[*] Đang tải dữ liệu WordNet...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def main():
    target_word = "bank"
    print("=" * 80)
    print(f"BÀI 1: TRA CỨU THÔNG TIN TỪ '{target_word}' TRONG WORDNET")
    print("=" * 80)

    # a. Liệt kê tất cả các synset của từ "bank"
    synsets = wn.synsets(target_word)
    print(f"\na. Liệt kê các synsets của từ '{target_word}':")
    print(f"   -> Tổng số synsets tìm thấy: {len(synsets)}")
    synset_names = [s.name() for s in synsets]
    print(f"   -> Danh sách tên synsets: {', '.join(synset_names)}")

    # b. Với mỗi synset, in ra tên, định nghĩa và ví dụ
    print(f"\nb. Chi tiết từng synset:")
    print("-" * 80)

    for idx, syn in enumerate(synsets, 1):
        name = syn.name()
        pos = syn.pos()  # Từ loại: n (danh từ), v (động từ), a (tính từ), r (trạng từ)
        pos_str = {
            'n': 'Danh từ (Noun)',
            'v': 'Động từ (Verb)',
            'a': 'Tính từ (Adjective)',
            's': 'Tính từ bổ trợ (Adjective Satellite)',
            'r': 'Trạng từ (Adverb)'
        }.get(pos, pos)
        definition = syn.definition()
        examples = syn.examples()

        print(f"[{idx}] Tên synset : {name} ({pos_str})")
        print(f"    Định nghĩa : {definition}")
        if examples:
            print(f"    Ví dụ ({len(examples)} ví dụ):")
            for ex in examples:
                print(f"      - \"{ex}\"")
        else:
            print(f"    Ví dụ      : (Không có câu ví dụ minh họa)")
        print("-" * 80)


if __name__ == "__main__":
    main()