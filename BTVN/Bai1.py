"""
BÀI TẬP: XÂY DỰNG MA TRẬN ĐỒNG XUẤT HIỆN CỦA TỪ (WORD CO-OCCURRENCE MATRIX)
Học phần: Xử lý Ngôn ngữ Tự nhiên (NLP)
Dữ liệu: testing.txt (hoặc Testing.txt)
"""

import os
import re
import unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class WordCoOccurrenceMatrix:
    def __init__(self, window_size=2, min_freq=5, max_vocab_size=100, weighting="flat"):
        """
        Khởi tạo bộ xây dựng ma trận đồng xuất hiện của từ.
        
        Tham số:
            window_size (int): Kích thước cửa sổ ngữ cảnh (mặc định 2: xét 2 từ trước và 2 từ sau).
            min_freq (int): Tần suất xuất hiện tối thiểu của từ để đưa vào từ điển (loại bỏ từ hiếm/nhiễu).
            max_vocab_size (int): Số lượng từ vựng tối đa đưa vào ma trận (để hiển thị và phân tích tối ưu).
            weighting (str): 
                - 'flat': Mỗi lần đồng xuất hiện cộng 1.
                - 'distance': Trọng số giảm dần theo khoảng cách (1 / khoảng_cách), theo chuẩn GloVe.
        """
        self.window_size = window_size
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.weighting = weighting

        self.vocab = []
        self.word2idx = {}
        self.idx2word = {}
        self.matrix = None
        self.word_freq = Counter()

    @staticmethod
    def preprocess_and_tokenize(text):
        """
        Tiền xử lý và tách từ cơ bản:
        1. Chuẩn hóa bảng mã Unicode (NFC)
        2. Loại bỏ nhãn danh mục như __CTXH__, __PL__, ...
        3. Chuyển thành chữ thường
        4. Tách các token từ (loại bỏ số, dấu câu đặc biệt)
        """
        # Chuẩn hóa Unicode NFC
        text = unicodedata.normalize("NFC", text)
        
        # Loại bỏ các tag danh mục dạng __TAG__ ở đầu văn bản
        text = re.sub(r"__[A-Z0-9_]+__", " ", text)
        
        # Loại bỏ đường link URL nếu có
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Tách từ: Lấy các chuỗi chữ cái tiếng Việt và số
        # [^\W\d_]+: chỉ lấy chữ cái (bao gồm chữ cái tiếng Việt có dấu), bỏ chữ số và ký tự đặc biệt
        tokens = re.findall(r"\b[^\W\d_]+\b", text, flags=re.UNICODE)
        
        # Lọc các từ có độ dài >= 2 ký tự (loại bỏ ký tự rác đơn lẻ)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens

    def fit(self, corpus_sentences):
        """
        Xây dựng tập từ vựng và tính toán ma trận đồng xuất hiện từ tập câu/văn bản.
        """
        print("[1/3] Đang thống kê tần suất từ vựng từ corpus...")
        # 1. Thống kê tần suất từ vựng trên toàn bộ corpus
        self.word_freq = Counter()
        tokenized_sentences = []
        
        for sentence in corpus_sentences:
            tokens = self.preprocess_and_tokenize(sentence)
            if tokens:
                tokenized_sentences.append(tokens)
                self.word_freq.update(tokens)

        # 2. Lọc từ điển theo min_freq và max_vocab_size
        filtered_words = [
            word for word, count in self.word_freq.most_common() 
            if count >= self.min_freq
        ]

        if self.max_vocab_size and len(filtered_words) > self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]

        self.vocab = sorted(filtered_words)
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}
        vocab_size = len(self.vocab)

        print(f"-> Tổng số token xử lý: {sum(self.word_freq.values()):,}")
        print(f"-> Số từ vựng duy nhất tìm thấy: {len(self.word_freq):,}")
        print(f"-> Kích thước từ điển chọn vào ma trận: {vocab_size} từ (Top {self.max_vocab_size} phổ biến nhất)")

        # 3. Khởi tạo ma trận rỗng V x V
        print(f"[2/3] Đang xây dựng ma trận đồng xuất hiện kích thước {vocab_size} x {vocab_size} (cửa sổ = {self.window_size})...")
        self.matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        # 4. Trượt cửa sổ và đếm đồng xuất hiện
        for tokens in tokenized_sentences:
            sent_len = len(tokens)
            for i, target_word in enumerate(tokens):
                if target_word not in self.word2idx:
                    continue
                target_idx = self.word2idx[target_word]

                # Xác định biên của cửa sổ trượt xung quanh từ mục tiêu
                left = max(0, i - self.window_size)
                right = min(sent_len, i + self.window_size + 1)

                for j in range(left, right):
                    if i == j:
                        continue  # Bỏ qua chính từ mục tiêu
                    
                    context_word = tokens[j]
                    if context_word in self.word2idx:
                        context_idx = self.word2idx[context_word]

                        # Tính trọng số đồng xuất hiện
                        if self.weighting == "distance":
                            weight = 1.0 / abs(i - j)
                        else:
                            weight = 1.0

                        self.matrix[target_idx, context_idx] += weight

        print("[3/3] Hoàn tất xây dựng ma trận đồng xuất hiện!")
        return self

    def to_dataframe(self, words=None):
        """
        Chuyển đổi ma trận sang định dạng pandas DataFrame.
        Nếu truyền danh sách words, chỉ lấy ma trận con cho các từ này.
        """
        if words is None:
            return pd.DataFrame(self.matrix, index=self.vocab, columns=self.vocab)
        
        # Chỉ lấy danh sách từ được yêu cầu
        valid_words = [w for w in words if w in self.word2idx]
        indices = [self.word2idx[w] for w in valid_words]
        sub_matrix = self.matrix[np.ix_(indices, indices)]
        return pd.DataFrame(sub_matrix, index=valid_words, columns=valid_words)

    def get_top_cooccurring(self, word, top_n=10):
        """
        Tìm các từ đồng xuất hiện nhiều nhất với một từ mục tiêu.
        """
        word = word.lower()
        if word not in self.word2idx:
            return f"Từ '{word}' không nằm trong từ điển được chọn."

        word_idx = self.word2idx[word]
        row = self.matrix[word_idx]

        # Sắp xếp chỉ số giảm dần theo điểm đồng xuất hiện
        top_indices = np.argsort(row)[::-1][:top_n]
        results = [
            (self.idx2word[idx], row[idx])
            for idx in top_indices
            if row[idx] > 0
        ]
        return results

    def plot_heatmap(self, words, title="Ma trận đồng xuất hiện của các từ chọn lọc", save_path=None):
        """
        Vẽ biểu đồ nhiệt (Heatmap) ma trận đồng xuất hiện cho một tập từ con.
        """
        valid_words = [w for w in words if w in self.word2idx]
        if not valid_words:
            print("Không tìm thấy từ nào hợp lệ trong tập từ điển để vẽ biểu đồ!")
            return

        df_sub = self.to_dataframe(valid_words)

        plt.figure(figsize=(10, 8))
        plt.imshow(df_sub.values, cmap="YlGnBu", interpolation="nearest")
        plt.colorbar(label="Tần suất đồng xuất hiện")
        plt.xticks(range(len(valid_words)), valid_words, rotation=45, ha="right", fontsize=10)
        plt.yticks(range(len(valid_words)), valid_words, fontsize=10)
        plt.title(title, fontsize=13, fontweight="bold", pad=15)
        
        # Hiển thị số liệu trực tiếp trên từng ô
        max_val = df_sub.values.max() if df_sub.values.max() > 0 else 1
        for i in range(len(valid_words)):
            for j in range(len(valid_words)):
                val = int(df_sub.values[i, j]) if df_sub.values[i, j].is_integer() else round(df_sub.values[i, j], 1)
                text_color = "white" if df_sub.values[i, j] > max_val / 2 else "black"
                plt.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Đã lưu biểu đồ nhiệt tại: {save_path}")
        plt.show()


def load_corpus(file_path):
    """
    Đọc dữ liệu từ file văn bản.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==============================================================================
if __name__ == "__main__":
    # Xác định đường dẫn file dữ liệu testing.txt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_filename = "Testing.txt"
    data_path = os.path.join(current_dir, data_filename)

    # Nếu không thấy Testing.txt thì tìm testing.txt (chữ thường)
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, "testing.txt")

    print("=" * 70)
    print("XÂY DỰNG MA TRẬN ĐỒNG XUẤT HIỆN CỦA TỪ (WORD CO-OCCURRENCE MATRIX)")
    print(f"Nguồn dữ liệu: {data_path}")
    print("=" * 70)

    # 1. Đọc corpus
    corpus = load_corpus(data_path)
    print(f"-> Đã nạp thành công {len(corpus):,} dòng văn bản từ corpus.\n")

    # 2. Khởi tạo và xây dựng ma trận
    # Cửa sổ ngữ cảnh window_size = 2 (xét 2 từ trước và 2 từ sau)
    # Lấy top 50 từ phổ biến nhất để ma trận trực quan, dễ quan sát
    WINDOW_SIZE = 2
    TOP_VOCAB_SIZE = 50

    cooc_model = WordCoOccurrenceMatrix(
        window_size=WINDOW_SIZE,
        min_freq=10,
        max_vocab_size=TOP_VOCAB_SIZE,
        weighting="flat"
    )
    cooc_model.fit(corpus)

    # 3. Hiển thị một góc ma trận (ví dụ 10 từ đầu tiên)
    print("\n" + "=" * 70)
    print("MẪU MA TRẬN ĐỒNG XUẤT HIỆN (10 từ đầu tiên trong từ điển):")
    print("=" * 70)
    sample_words = cooc_model.vocab[:10]
    sample_df = cooc_model.to_dataframe(sample_words)
    print(sample_df.to_string())

    # 4. Xuất ma trận đầy đủ ra file CSV
    output_csv_path = os.path.join(current_dir, "cooccurrence_matrix.csv")
    full_df = cooc_model.to_dataframe()
    full_df.to_csv(output_csv_path, encoding="utf-8-sig")
    print(f"\n-> Đã xuất ma trận ({len(cooc_model.vocab)}x{len(cooc_model.vocab)}) ra file: {output_csv_path}")

    # 5. Tra cứu các từ đồng xuất hiện nhiều nhất với một số từ tiêu biểu
    test_keywords = ["người", "thành", "phố", "việt", "nam", "quốc", "nước", "hàng"]
    print("\n" + "=" * 70)
    print("TRA CỨU TOP CÁC TỪ ĐỒNG XUẤT HIỆN NHIỀU NHẤT VỚI TỪ KHÓA:")
    print("=" * 70)
    for kw in test_keywords:
        if kw in cooc_model.word2idx:
            top_cooc = cooc_model.get_top_cooccurring(kw, top_n=5)
            formatted = ", ".join([f"'{w}': {int(cnt)}" for w, cnt in top_cooc])
            print(f"- Từ '{kw}': {formatted}")

    # 6. Vẽ biểu đồ nhiệt (Heatmap) cho 15 từ tiêu biểu
    selected_words_for_plot = [w for w in test_keywords if w in cooc_model.word2idx]
    # Lấy thêm các từ khác trong top từ vựng cho đủ 15 từ
    for w in cooc_model.vocab:
        if w not in selected_words_for_plot:
            selected_words_for_plot.append(w)
        if len(selected_words_for_plot) >= 15:
            break

    plot_image_path = os.path.join(current_dir, "cooccurrence_heatmap.png")
    print(f"\n-> Đang tạo biểu đồ nhiệt cho 15 từ: {selected_words_for_plot}")
    cooc_model.plot_heatmap(
        words=selected_words_for_plot,
        title=f"Ma trận đồng xuất hiện giữa các từ (Cửa sổ = {WINDOW_SIZE})",
        save_path=plot_image_path
    )
    print("\nCHƯƠNG TRÌNH HOÀN THÀNH!")
