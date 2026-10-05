# BÀI TẬP LẬP TRÌNH NLP - THƯ VIỆN NLTK, WORDNET & WORD2VEC

Thư mục `BaiKtra/` chứa toàn bộ lời giải hoàn chỉnh, độc lập và chi tiết cho **12 bài tập** trong tệp đề bài `Bài kiểm tra NLP.pdf`.

---

## Danh mục các bài tập và tệp thực thi

| Bài tập | Tên file | Mô tả & Nội dung chính |
| :--- | :--- | :--- |
| **Bài 1** | [`Bai1.py`](file:///d:/NLP_Tina4/BaiKtra/Bai1.py) | **Tra cứu thông tin từ trong WordNet**: Liệt kê tất cả các synset của từ `"bank"`, in ra tên, từ loại (POS), định nghĩa (definition) và các ví dụ (examples). |
| **Bài 2** | [`Bai2.py`](file:///d:/NLP_Tina4/BaiKtra/Bai2.py) | **Phân tích cấu trúc quan hệ WordNet**: Lấy synset `"car.n.01"` và truy vấn các quan hệ ngữ nghĩa: Hypernyms (bao hàm), Hyponyms (phân nhánh), Meronyms (bộ phận), Holonyms (toàn thể). |
| **Bài 3** | [`Bai3.py`](file:///d:/NLP_Tina4/BaiKtra/Bai3.py) | **Tìm nghĩa gần đúng của từ trong câu**: Với câu *"The crane is standing near the river."*, trích xuất từ `"crane"`, liệt kê các nghĩa có thể và dự đoán nghĩa phù hợp nhất (`crane.n.05` - chim sếu thay vì máy cẩu). |
| **Bài 4** | [`Bai4.py`](file:///d:/NLP_Tina4/BaiKtra/Bai4.py) | **Độ tương đồng bằng Path Similarity**: Tính độ đo $\frac{1}{\text{path\_distance} + 1}$ giữa các cặp từ: `car - automobile`, `dog - cat`, `dog - wolf`, `tree - car` và nhận xét kết quả. |
| **Bài 5** | [`Bai5.py`](file:///d:/NLP_Tina4/BaiKtra/Bai5.py) | **Độ tương đồng bằng Wu-Palmer Similarity**: Sử dụng `wup_similarity` dựa trên độ sâu nút tổ tiên chung gần nhất (LCS) và so sánh chuyên sâu với `path_similarity`. |
| **Bài 6** | [`Bai6.py`](file:///d:/NLP_Tina4/BaiKtra/Bai6.py) | **Tính độ tương đồng từ vựng dựa trên WordNet**: Xây dựng hàm `wordnet_similarity(word1, word2)` tìm max WUP giữa mọi cặp synset; kiểm tra với `ship - boat`, `good - bad`, `food - fruit`, `computer - laptop`. |
| **Bài 7** | [`Bai7.py`](file:///d:/NLP_Tina4/BaiKtra/Bai7.py) | **Ứng dụng phân loại cảm xúc đơn giản**: Nhập từ tiếng Anh bất kỳ, tính độ tương đồng với tập `Positive` và `Negative`, gán nhãn cảm xúc cho từ. Hỗ trợ cả tương tác trực tiếp lẫn chạy tự động. |
| **Bài 8** | [`Bai8.py`](file:///d:/NLP_Tina4/BaiKtra/Bai8.py) | **Phân cụm từ vựng bằng độ đo tương đồng**: Tính ma trận WUP $8 \times 8$ cho 8 từ vựng, áp dụng K-means ($k=2$) phân tách hoàn hảo 100% nhóm Động vật (`cat`, `dog`, `wolf`, `lion`) và nhóm Phương tiện (`car`, `truck`, `bus`, `airplane`). |
| **Bài 9** | [`Bai9.py`](file:///d:/NLP_Tina4/BaiKtra/Bai9.py) | **Tính Semantic Similarity cho câu**: Xây dựng hàm `sentence_similarity(sent1, sent2)` tính độ tương đồng trung bình max giữa các từ, so sánh 3 câu mẫu và phân tích ý nghĩa. |
| **Bài 10** | [`Bai10.py`](file:///d:/NLP_Tina4/BaiKtra/Bai10.py) | **Khử nhập nhằng ngữ nghĩa bằng thuật toán Lesk (WSD)**: Khử nhập nhằng cho từ `"bass"` trong hai ngữ cảnh âm nhạc (*"I can hear the bass in this music."*) và câu cá (*"He caught a huge bass in the lake."*). |
| **Bài 11** | [`Bai11.py`](file:///d:/NLP_Tina4/BaiKtra/Bai11.py) | **Tìm từ tương đồng bằng Word2Vec**: Trích xuất top-5 từ tương tự nhất cho `"dog"` và `"cat"`, phân tích nguyên lý biểu diễn phân bố ngữ nghĩa (Distributional Hypothesis). |
| **Bài 12** | [`Bai12.py`](file:///d:/NLP_Tina4/BaiKtra/Bai12.py) | **Cosine Similarity Word2Vec & So sánh đối chiếu WordNet**: Xây dựng hàm `cosine_sim(word1, word2, model)` tính tương đồng cho `dog - cat`, `dog - car`, `king - queen`, đối chiếu toàn diện với WordNet. |
| **Tất cả** | [`run_all.py`](file:///d:/NLP_Tina4/BaiKtra/run_all.py) | **Script chạy tự động toàn bộ 12 bài tập**: Chạy một lệnh duy nhất để xuất kết quả của cả 12 bài tập. |

---

## Hướng dẫn chạy

### Chạy từng bài riêng lẻ:
```bash
python BaiKtra/Bai1.py
python BaiKtra/Bai2.py
python BaiKtra/Bai3.py
python BaiKtra/Bai4.py
python BaiKtra/Bai5.py
python BaiKtra/Bai6.py
python BaiKtra/Bai7.py
python BaiKtra/Bai8.py
python BaiKtra/Bai9.py
python BaiKtra/Bai10.py
python BaiKtra/Bai11.py
python BaiKtra/Bai12.py
```

### Chạy đồng loạt cả 12 bài:
```bash
python BaiKtra/run_all.py
```

---

## Tóm tắt kiến thức lý thuyết

1. **WordNet**: Mạng ngữ nghĩa (lexical database) tổ chức từ vựng thành các tập hợp đồng nghĩa (*synsets*). Các mối quan hệ chính:
   - **Hypernym / Hyponym**: Quan hệ loại - thể hiện cấu trúc IS-A (cây phả hệ).
   - **Meronym / Holonym**: Quan hệ bộ phận - toàn thể (Part-of).
2. **Path Similarity vs Wu-Palmer Similarity**:
   - $\text{Path\_Sim}(s_1, s_2) = \frac{1}{\text{shortest\_path} + 1}$
   - $\text{WUP\_Sim}(s_1, s_2) = \frac{2 \times \text{depth}(\text{LCS})}{\text{depth}(s_1) + \text{depth}(s_2)}$ (tính đến độ sâu của tổ tiên chung gần nhất).
3. **Word Sense Disambiguation (Thuật toán Lesk)**:
   - Đếm số lượng từ trùng khớp (*overlap*) giữa từ ngữ cảnh trong câu và định nghĩa + ví dụ của từng synset ứng viên trong từ điển.
4. **Word2Vec vs WordNet**:
   - **WordNet**: Tri thức hình thức do chuyên gia gán nhãn (Taxonomy), tính minh bạch cao nhưng khó mở rộng.
   - **Word2Vec**: Học biểu diễn vector phân bố không giám sát từ kho ngữ liệu khổng lồ (Distributional semantics), nắm bắt linh hoạt quan hệ ngữ cảnh thực tế và suy luận loại suy (*analogy*).
