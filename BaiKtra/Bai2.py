
import nltk
from nltk.corpus import wordnet as wn

# Đảm bảo tải tài nguyên WordNet
try:
    wn.synset("car.n.01")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def format_synsets(synset_list):
    """Định dạng danh sách synset thành chuỗi dễ đọc kèm định nghĩa ngắn."""
    if not synset_list:
        return ["  (Không có dữ liệu trong WordNet)"]
    lines = []
    for s in synset_list:
        lines.append(f"  - {s.name()}: {s.definition()}")
    return lines


def main():
    print("=" * 80)
    print("BÀI 2: PHÂN TÍCH CẤU TRÚC QUAN HỆ WORDNET CHO 'car.n.01'")
    print("=" * 80)

    # a. Lấy synset "car.n.01"
    syn = wn.synset("car.n.01")
    print("\na. Thông tin synset được chọn:")
    print(f"   - Tên synset : {syn.name()}")
    print(f"   - Định nghĩa : {syn.definition()}")
    print(f"   - Lemmas     : {[l.name() for l in syn.lemmas()]}")
    print(f"   - Ví dụ      : {syn.examples()}")

    # b. In ra danh sách các quan hệ
    print("\nb. Phân tích các mối quan hệ ngữ nghĩa trong WordNet:")
    print("=" * 80)

    # 1. Hypernyms (Quan hệ bao hàm: A là cha của B - B IS-A A)
    hypernyms = syn.hypernyms()
    print(f"\n1. HYPERNYMS (Quan hệ bao hàm / Cấp trên - {len(hypernyms)} mục):")
    print("   (Ý nghĩa: 'car' là một loại của ...)")
    for line in format_synsets(hypernyms):
        print(line)

    # Thêm Root Hypernyms để xem gốc phả hệ
    root_hypernyms = syn.root_hypernyms()
    print(f"   -> Gốc cao nhất (Root): {[r.name() for r in root_hypernyms]}")

    # 2. Hyponyms (Quan hệ phân nhánh: B là con của A - B là loại cụ thể của car)
    hyponyms = syn.hyponyms()
    print(f"\n2. HYPONYMS (Quan hệ phân nhánh / Cấp dưới - {len(hyponyms)} mục):")
    print("   (Ý nghĩa: Các loại xe hơi cụ thể ...)")
    for line in format_synsets(hyponyms):
        print(line)

    # 3. Meronyms (Quan hệ bộ phận: Part/Substance/Member cấu thành nên car)
    part_meronyms = syn.part_meronyms()
    substance_meronyms = syn.substance_meronyms()
    member_meronyms = syn.member_meronyms()

    print(f"\n3. MERONYMS (Quan hệ bộ phận - Các thành phần cấu tạo nên 'car'):")
    print(f"   * Part Meronyms (Bộ phận cấu thành - {len(part_meronyms)} mục):")
    for line in format_synsets(part_meronyms):
        print(f"  {line}")

    if substance_meronyms:
        print(f"   * Substance Meronyms (Chất liệu cấu thành - {len(substance_meronyms)} mục):")
        for line in format_synsets(substance_meronyms):
            print(f"  {line}")
    else:
        print("   * Substance Meronyms: (Không có)")

    if member_meronyms:
        print(f"   * Member Meronyms (Thành viên - {len(member_meronyms)} mục):")
        for line in format_synsets(member_meronyms):
            print(f"  {line}")
    else:
        print("   * Member Meronyms: (Không có)")

    # 4. Holonyms (Quan hệ toàn thể: 'car' là một phần/thành viên của thực thể lớn hơn nào)
    part_holonyms = syn.part_holonyms()
    substance_holonyms = syn.substance_holonyms()
    member_holonyms = syn.member_holonyms()

    print(f"\n4. HOLONYMS (Quan hệ toàn thể - 'car' thuộc về chỉnh thể lớn nào):")
    if part_holonyms:
        print(f"   * Part Holonyms ({len(part_holonyms)} mục):")
        for line in format_synsets(part_holonyms):
            print(f"  {line}")
    else:
        print("   * Part Holonyms: (Không có)")

    if substance_holonyms:
        print(f"   * Substance Holonyms ({len(substance_holonyms)} mục):")
        for line in format_synsets(substance_holonyms):
            print(f"  {line}")
    else:
        print("   * Substance Holonyms: (Không có)")

    if member_holonyms:
        print(f"   * Member Holonyms ({len(member_holonyms)} mục):")
        for line in format_synsets(member_holonyms):
            print(f"  {line}")
    else:
        print("   * Member Holonyms: (Không có)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()