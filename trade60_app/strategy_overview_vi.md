# Tổng quan chiến lược Trade60

## Mục tiêu

Trade60 là hệ thống giao dịch cổ phiếu **chỉ mua, dùng tiền mặt, không margin** cho tập `60` mã cổ phiếu đã huấn luyện. Hệ thống tìm cách:

- vượt mục tiêu lợi nhuận tối thiểu `8%/năm`
- cố gắng vượt `VNINDEX`
- hạn chế rút vốn quá sâu trong giai đoạn xấu

## Dữ liệu và phạm vi

- Dữ liệu giá ngày được lấy từ `vnstock`
- Hệ thống cập nhật bổ sung phần dữ liệu còn thiếu thay vì tải lại toàn bộ
- Thời gian nghiên cứu: từ `2006` đến hiện tại
- Tần suất ra quyết định: **đưa lệnh cho phiên tiếp theo**

Hệ thống này **không phải giao dịch intraday**. Nó dùng dữ liệu ngày và đưa ra khuyến nghị mua, bán, giảm tỷ trọng hoặc giữ nguyên cho ngày kế tiếp.

## Cách chấm điểm cổ phiếu

Mô hình sử dụng các nhóm đặc trưng chính:

- động lượng ngắn, trung hạn
- độ mạnh tương đối so với `VNINDEX`
- xu hướng so với MA
- biến động và thanh khoản
- độ rộng thị trường của rổ 60 mã

Mỗi mã sẽ có:

- `alpha_probability`: độ tự tin của cổ phiếu
- `regime_probability`: độ thuận lợi của môi trường thị trường chung
- `composite_score`: điểm tổng hợp để xếp hạng

## Nguyên tắc quản trị rủi ro

Hệ thống có các quy tắc quan trọng:

- không mua và bán cùng một mã trong cùng ngày
- không giữ một mã quá `40` phiên giao dịch
- có stop-loss và take-profit
- khi thị trường yếu, hệ thống tự giảm mức giải ngân và giữ thêm tiền mặt

## Cách backtest

Hệ thống được đánh giá theo hai lớp:

- **validation walk-forward** để chọn tham số trên các giai đoạn lịch sử
- **final test untouched** là giai đoạn cuối không dùng để tune tham số

Bạn nên ưu tiên nhìn vào:

- `annualized_return`
- `excess_return_vs_benchmark`
- `max_drawdown`
- `sharpe`

## Cách đọc khuyến nghị hằng ngày

Màn hình sẽ nhận:

- ngân sách tiền mặt còn lại
- số lượng đang nắm giữ theo từng mã
- giá vốn
- ngày mua

Sau đó hệ thống trả về một trong các kiểu hành động:

- `MUA`: đề xuất mua trong phiên tới
- `BÁN`: đề xuất bán trong phiên tới
- `ĐỨNG NGOÀI`: tạm thời không mở vị thế mới

Nếu bạn đã có hàng sẵn, màn hình còn hiển thị thêm **trạng thái danh mục** để phân biệt rõ:

- giữ nguyên vị thế hiện tại
- giảm một phần vị thế
- thoát hết vị thế
- tăng thêm vị thế đang có

Khuyến nghị là công cụ hỗ trợ quyết định, không phải cam kết lợi nhuận.
