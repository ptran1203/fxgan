

## Data augmentation for chestx-ray classification using GAN

### Abstract
- Huấn luyện mô hình học sâu (Deep learning) trong bài toán phân lớp (hoặc phát hiện) bệnh lý trong ảnh y khoa hiện tại còn gặp nhiều hạn chế do dữ liệu thường có xu hướng bị mất cân bằng, điều này dẫn đến mô hình thường thiên vị về những lớp có nhiều dữ liệu hơn và việc dự đoán những lớp có số lượng dữ liệu ít sẽ có độ chính xác không cao. Chúng tôi xây dựng mô hình Generative adversarial network để tăng cường dữ liệu để cải thiện quá trình huấn luyện mô hình Deep learning.

### 1. Introduction
- Phân loại ảnh y khoa là một thành phần quan trọng trong những hệ thống hỗ trợ chuẩn đoán bệnh lý, tiếp cận bằng kỹ thuật deep learning đã tạo ra những mô hình chuẩn đoán có hiệu quả cao. Tuy nhiên để huấn luyện mô hình deep learning cần một lượng dữ liệu được gán nhãn đủ lớn, đây là một thách thức trong lĩnh vực y khoa vì:
1) Chính sách bảo mật
2) Yêu cầu chuyên gia về bệnh lý cụ thể để gán nhãn
3) Dữ liệu ít ỏi của một số bệnh hiếm có dẫn đến mất cân bằng
Trong bài báo này, chúng tôi sử dụng bộ dữ liệu Chest x-ray14 để đánh giá hướng tiếp cận tăng cường dữ liệu sử dụng GAN. Đây là bộ dữ liệu công khai bao gồm 14 bệnh lý liên quan đến phổi, đặc biệt bộ dữ liệu này bị mất cân bằng giữa các lớp, trong đó lớp normal (không mang bệnh) chiếm xấp xỉ 50% của toàn bộ dữ liệu, trong khi đó ở các lớp mang bệnh thì lớp chiếm tỉ lệ cao nhất là 20% và thấp nhất chỉ 2%. Trong nghiên cứu này chúng tôi sẽ không thực hiện trên những dữ liệu mang nhiều bệnh (multi-label) mà chỉ sử dụng những ảnh mang đúng một bệnh lý.

- Huấn luyện trên những bộ dữ liệu bị mất cân bằng thường sẽ không đạt được chất lượng tốt. Thường thì mô hình sẽ thiên vị về những lớp mang nhiều dữ liệu hơn để tối ưu hàm mất mát, dẫn đến việc dự đoán những lớp có ít dữ liệu không được chính xác. Phương pháp tăng cường dữ liệu (Data augmentation) có thể được dùng để giải quyết vấn đề này, Trong thực tế các kỹ thuật tăng cường dữ liệu điển hình sử dụng các phép biến đổi lên ảnh hiện có, dẫn đến sự đa dạng về đặc tính của ảnh còn hạn chế
## Những thực nghiệm

Dataset:
 + Trong