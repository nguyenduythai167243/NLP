"""
SCRIPT CHẠY TỰ ĐỘNG TẤT CẢ 12 BÀI TẬP KIỂM TRA NLP
Chạy lần lượt từ Bài 1 đến Bài 12 trong thư mục BaiKtra
"""

import sys
import os
import subprocess

EXERCISES = [
    ("Bài 1", "Bai1.py", "Tra cứu thông tin từ trong WordNet"),
    ("Bài 2", "Bai2.py", "Phân tích cấu trúc quan hệ WordNet"),
    ("Bài 3", "Bai3.py", "Tìm nghĩa gần đúng của từ trong câu"),
    ("Bài 4", "Bai4.py", "Độ tương đồng bằng Path Similarity"),
    ("Bài 5", "Bai5.py", "Độ tương đồng bằng Wu-Palmer Similarity"),
    ("Bài 6", "Bai6.py", "Tính độ tương đồng từ vựng dựa trên WordNet"),
    ("Bài 7", "Bai7.py", "Ứng dụng phân loại cảm xúc đơn giản dùng Word Similarity"),
    ("Bài 8", "Bai8.py", "Phân cụm từ vựng bằng độ đo tương đồng & K-means"),
    ("Bài 9", "Bai9.py", "Tính semantic similarity cho câu"),
    ("Bài 10", "Bai10.py", "Khử nhập nhằng ngữ nghĩa bằng thuật toán Lesk"),
    ("Bài 11", "Bai11.py", "Tìm từ similar bằng Word2Vec"),
    ("Bài 12", "Bai12.py", "Tính độ tương đồng giữa từ bằng Cosine Similarity & So sánh"),
]


def run_exercise(name, filename, title, py_executable):
    print("\n" + "#" * 90)
    print(f"# {name.upper()}: {title.upper()} (File: {filename})")
    print("#" * 90 + "\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    try:
        proc = subprocess.run(
            [py_executable, file_path],
            cwd=current_dir,
            text=True,
            capture_output=False
        )
        if proc.returncode != 0:
            print(f"[!] {name} kết thúc với mã lỗi: {proc.returncode}")
    except Exception as e:
        print(f"[!] Lỗi khi chạy {filename}: {e}")


def main():
    py_executable = sys.executable
    print("=" * 90)
    print("CHƯƠNG TRÌNH TỰ ĐỘNG CHẠY TOÀN BỘ 12 BÀI TẬP KIỂM TRA NLP")
    print(f"Trình thông dịch Python đang dùng: {py_executable}")
    print("=" * 90)

    for name, filename, title in EXERCISES:
        run_exercise(name, filename, title, py_executable)

    print("\n" + "=" * 90)
    print("HOÀN TẤT CHẠY 12/12 BÀI TẬP TRONG BÀI KIỂM TRA NLP!")
    print("=" * 90)


if __name__ == "__main__":
    main()
