# Introduction



Kỹ thuật Deep learning được áp dụng rộng rãi trong y khoa, đặc biệt với sự phát triển của công nghệ tính toán như GPU và các thuật toán hiện đại với độ chính xác cao. Tuy nhiên các mô hình Deep learning thường yêu cầu số lượng lớn dữ liệu được gán nhãn để huấn luyện (supervised learning). Trong khi đó, trong lĩnh vực y khoa, chính sách bảo mật thông tin bệnh nhân dẫn đến việc truy cập vào dữ liệu rất khó khăn, điều này đẫn đến những bộ dữ liệu thường có số lượng không nhiều, đặc biệt đối với những căn bệnh hiếm gặp thì dữ liệu sẽ càng ít. Trong nghiên cứu này, chúng tôi sử dụng ảnh x-quang ngực làm bộ dữ liệu để thực hiện bài toán phân lớp (classification).

Trong trường hợp bộ dữ liệu có số lượng nhỏ và đôi khi mất cân bằng, nó có thể được đáp ứng một phần dựa vào kỹ thuật tăng cường dữ liệu (Data augmentation), bằng cách áp dụng các phép biến đổi hình học lên những ảnh hiện có để tăng số lượng ảnh. Tuy nhiên nó chưa đủ đáp ứng được.