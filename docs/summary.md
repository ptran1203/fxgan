
# Data augmentation for chestx-ray classification using GAN

## Abstract

 Huấn luyện mô hình học sâu (Deep learning) trong bài toán phân lớp (hoặc phát hiện) bệnh lý trong ảnh y khoa hiện tại còn gặp nhiều hạn chế do dữ liệu thường có xu hướng bị mất cân bằng, điều này dẫn đến mô hình thường thiên vị về những lớp có nhiều dữ liệu hơn và việc dự đoán những lớp có số lượng dữ liệu ít sẽ có độ chính xác không cao. Chúng tôi xây dựng mô hình Generative adversarial network để tăng cường dữ liệu để cải thiện quá trình huấn luyện mô hình Deep learning.

## 1. Introduction

 Phân loại ảnh y khoa là một thành phần quan trọng trong những hệ thống hỗ trợ chuẩn đoán bệnh lý, tiếp cận bằng kỹ thuật deep learning đã tạo ra những mô hình chuẩn đoán có hiệu quả cao. Tuy nhiên để huấn luyện mô hình deep learning cần một lượng dữ liệu được gán nhãn đủ lớn, đây là một thách thức trong lĩnh vực y khoa vì:

1) Chính sách bảo mật
2) Yêu cầu chuyên gia về bệnh lý cụ thể để gán nhãn
3) Dữ liệu ít ỏi của một số bệnh hiếm có dẫn đến mất cân bằng
 Trong bài báo này, chúng tôi sử dụng bộ dữ liệu Chest x-ray14 để đánh giá hướng tiếp cận tăng cường dữ liệu sử dụng GAN. Đây là bộ dữ liệu công khai bao gồm 14 bệnh lý liên quan đến phổi, đặc biệt bộ dữ liệu này bị mất cân bằng giữa các lớp, trong đó lớp normal (không mang bệnh) chiếm xấp xỉ 50% của toàn bộ dữ liệu, trong khi đó ở các lớp mang bệnh thì lớp chiếm tỉ lệ cao nhất là 20% và thấp nhất chỉ 2%. Trong nghiên cứu này chúng tôi sẽ không thực hiện trên những dữ liệu mang nhiều bệnh (multi-label) mà chỉ sử dụng những ảnh mang đúng một bệnh lý.

 Huấn luyện trên những bộ dữ liệu bị mất cân bằng thường sẽ không đạt được chất lượng tốt. Thường thì mô hình sẽ thiên vị về những lớp mang nhiều dữ liệu hơn để tối ưu hàm mất mát, dẫn đến việc dự đoán những lớp có ít dữ liệu không được chính xác. Phương pháp tăng cường dữ liệu (Data augmentation) có thể được dùng để giải quyết vấn đề này, Trong thực tế các kỹ thuật tăng cường dữ liệu điển hình sử dụng các phép biến đổi lên ảnh hiện có, dẫn đến sự đa dạng về đặc tính của ảnh còn hạn chế. Một số nghiên cứu đã chứng minh rằng GAN có thể tổng hợp được ảnh mang đặc tính cần thiết để huấn luyện các mô hình DL [list the works here]. Trong nghiên cứu này chúng tôi sử dụng GAN để tổng hợp ảnh x-ray ngực dựa trên mô hình conditional Generative Adversarial Network (cGAN).

Chúng tôi chọn mô hình BAGAN và mô hình dựa trên openGAN vì nó thích hợp cho bộ dữ liệu mất cân bằng và ảnh được tạo ra mang thông tin của bệnh cụ thể. Chúng tôi đánh giá kết quả của hai phương pháp trên bài toán phân lớp ảnh chestxray ngực để phát hiện bệnh về phổi. Chứng minh khả năng cải thiện hiệu năng của mô hình phân lớp với lượng dữ liệu nhỏ hơn.

## 2. Phương pháp

 ### 2.1 BAGAN
Tóm tắt: BAGAN tận dụng autoencoder để giúp Generator học dễ dàng hơn. Cụ thể ta sẽ huấn luyện autoencoder trước sau đó tính toán multivariate distribution được lấy từ output của encoder, đồng thời khởi tạo trọng số cho Generator bằng trọng số của decoder. Khi train Generator latent vector sẽ được lấy từ multivariate distribution. Discriminator sẽ làm nhiệm vụ phân biệt ảnh input là ảnh giả hay ảnh thuộc một class nào đó, điều này giúp cho ảnh được tạo phải mang thông tin của một lớp cụ thể.
 ### 2.1 new GAN
Sử dụng feature normalization từ openGAN, chúng tôi thêm cho thêm thông tin của ảnh vào latent vector bằng cách concatenate noise z và  feature được trích từ  pre-trained VGG16 trên bộ dữ liệu đó.

Tóm tắt: Open GAN sử dụng feature normalization để mã hoá thông tin vào Generator, Feature sẽ được trích từ một pre-trained Deep metric classification. Với cách thêm thông tin vào G như trên sẽ giúp cho Generator tổng hợp được ảnh với unseen classes.


## 3. Thực nghiệm

### 3.1 BAGAN

Thực nghiệm trên bộ dữ liệu chestxray14 bao gồm 5 classes, kích thước ảnh được resize về 64*64 cho hiệu quả tính toán.
Chúng tôi thay thế neareast upsampling của BAGAN bằng DeConvolution layer (DCGAN) để giúp tăng chất lượng ảnh.
sau đó trong quá trình lấy mẫu, chúng tôi chỉ lấy những ảnh từ Generator thoã mãn điều kiện thông qua kết quả từ Discriminator.

### 3.2 open GAN:

Tương tự như BAGAN, chúng tôi huấn luyện trên 5 classes, kích thước giảm về 64*64. Sử dụng Generator tương tự như đề xuất của openGAN và Discriminator là những lớp Conv với strides =2 để downsampling.


## 4. Đánh giá
    Kết quả classification được đánh giá bằng AUC scores của mỗi class như sau:

### 4.1. AUC score on 5 classes (augment 1000 images per class)

| | VGG16 | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|--|
| No finding | 0.713 | 0.722 | **`0.726`** |0.723 |
|Infiltration| 0.702 | **`0.712`** | 0.699 | 0.706 |
|Atelectasis| 0.738 | 0.742| **`0.752`** | 0.746 |
|Effusion| 0.812 | **`0.818`** | 0.813 |0.814 |
|Nodule| 0.728 |**`0.735`**| 0.732 |0.719 |

### 4.2. AUC score on 5 classes: (augment 3000 images per class)

| | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|
| No finding | 0.725 | 0.712 | **`0.736`** |
|Infiltration| **`0.69`**  | 0.693 | 0.69 |
|Atelectasis| 0.735 | **`0.747`** | 0.744|
|Effusion| **`0.814`** | 0.81 |0.813|
|Nodule| 0.728 |0.725| **`0.737`** |


BAGAN mang nhiều thông tin của một lớp hơn so với newGAN nhờ vào output của Discriminator giúp Generator tạo ra ảnh chính xác hơn.