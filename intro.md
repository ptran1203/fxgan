# Introduction



Kỹ thuật học sâu (Deep learning) [1] đã được áp dụng rộng rãi trong y khoa, đặc biệt với sự phát triển của công nghệ tính toán như GPU và các thuật toán hiện đại với độ chính xác cao. Tuy nhiên các mô hình Deep learning thường yêu cầu số lượng lớn dữ liệu được gán nhãn để huấn luyện (supervised learning). Trong khi đó, trong lĩnh vực y khoa, chính sách bảo mật thông tin bệnh nhân dẫn đến việc truy cập vào dữ liệu rất khó khăn [2], điều này đẫn đến những bộ dữ liệu thường có số lượng không nhiều, đặc biệt đối với những căn bệnh hiếm gặp thì dữ liệu sẽ càng ít. Trong nghiên cứu này, chúng tôi sử dụng ảnh X-quang ngực làm bộ dữ liệu để thực hiện bài toán phân lớp (classification) [3].

Trong trường hợp bộ dữ liệu có số lượng nhỏ và đôi khi mất cân bằng [4], nó có thể được đáp ứng một phần dựa vào kỹ thuật tăng cường dữ liệu (Data augmentation) [5], bằng cách áp dụng các phép biến đổi hình học lên những ảnh hiện có để tăng số lượng ảnh. Tuy nhiên phương pháp này chỉ sản sinh một lượng ảnh nhất định và những trường hợp cụ thể. Mạng sạng tạo đối nghịch (Generative Adversarial Network - GAN) [6] đã chứng tỏ khả năng sản sinh dữ liệu nhân tạo một cách hiểu quả, một số bài báo đã chứng minh rằng dữ liệu được tổng hợp từ GAN đã góp phần cải thiện hiệu suất của các bài toán phân lớp (classification)và phân đoạn ảnh (image segmentation) [7,8,9]. Đặc biệt đối với các bài toán về y khoa [10,11,12]

Trong nghiên cứu này, chúng tôi giải quyết trường hợp thử thách hơn trong bài toán phân lớp các bệnh về phổi qua ảnh X-quang (chest X-ray classification) [13] khi mà một số class chỉ có số lượng nhỏ ảnh được gán nhãn (từ 0 đến 5 ảnh mỗi lớp), vấn đề này liên quan đến bài toán few-shot learning [14]. Chúng tôi tiếp cận bằng phương pháp sử dụng GAN sản sinh ra ảnh và sau đó tăng cường dữ liệu cho các mô hình few-shot learning như MatchingNet [15], RelationNet[16]. 

Chúng tôi đề xuất mô hình GAN có thể sản sinh ra ảnh giả dựa vào một vài ảnh Xs {s: 1->k} thuộc cùng một lớp [17], trong đó k là số lượng ảnh. Trong giai đoạn huấn luyện, GAN sẽ được huấn luyện với những lớp Cs (Seen category), sau đó trong giai đoạn thử nghiệm GAN sẽ được dùng để sản sinh ảnh dựa vào những ảnh từ những lớp Cu (Unseen category) mà GAN không được học trong qúa trình huấn luyện. Cụ thể về phương pháp sẽ được bàn luận chi tiết ở mục ...

Những đóng góp chính của chúng tôi trong nghiên cứu lần này bao gồm:
+ Xây dựng mô hình few-shot image generation
+ Thực hiện các thử nghiệp và so sánh hiệu suất dựa vào bộ dữ liệu Chest-xray
+ So sánh với các mô hình few-shot image generation khác thông qua các bộ dữ liệu tiêu chuẩn như EMNIST, ...