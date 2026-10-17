import { isTechnicalTerm } from './technical-terms';

const labels: Record<string, string> = {
  Start: 'Bắt đầu', Stop: 'Dừng', Play: 'Chạy', Pause: 'Tạm dừng', Reset: 'Đặt lại',
  Generate: 'Sinh token', 'Next token': 'Token tiếp theo', 'Input text': 'Văn bản đầu vào',
  Temperature: 'Nhiệt độ', 'Top K': 'Top K', 'Sampling': 'Lấy mẫu',
  Attention: 'Chú ý', 'Self-Attention': 'Tự chú ý', 'Embedding': 'Biểu diễn vector',
  'Text embedding': 'Biểu diễn văn bản', 'Text Encoder': 'Bộ mã hóa văn bản',
  'Image Decoder': 'Bộ giải mã ảnh', 'Latent representation': 'Biểu diễn tiềm ẩn',
  'Noise': 'Nhiễu', 'Seed': 'Seed', 'Random seed': 'Seed ngẫu nhiên',
  'Guidance scale': 'Mức định hướng', 'Prompt': 'Mô tả', 'Compare': 'So sánh',
  'Compare prompts': 'So sánh mô tả', 'Timestep': 'Bước',
  'Generator': 'Mạng sinh', 'Discriminator': 'Mạng phân biệt', 'Real data': 'Dữ liệu thật',
  'Generated data': 'Dữ liệu sinh', 'Learning rate': 'Tốc độ học', 'Batch size': 'Kích thước lô',
  'Iterations': 'Số vòng lặp', 'Iteration': 'Vòng lặp', 'Training': 'Huấn luyện',
  'Loss': 'Hàm mất mát', 'Noise dimension': 'Số chiều nhiễu',
  'Dataset': 'Bộ dữ liệu', 'Select dataset': 'Chọn bộ dữ liệu', 'Features': 'Đặc trưng',
  'Accuracy': 'Độ chính xác', 'Precision': 'Độ chuẩn xác', 'Recall': 'Độ bao phủ',
  'Subgroups': 'Nhóm con', 'Suggested groups': 'Nhóm được đề xuất',
  'Confusion Matrix': 'Ma trận nhầm lẫn', 'Neuron': 'Nơ-ron', 'Neurons': 'Các nơ-ron',
  'Underperforming': 'Hiệu suất thấp', 'Well-performing': 'Hiệu suất tốt',
  'Activation threshold': 'Ngưỡng kích hoạt', 'Prediction': 'Dự đoán',
  'Energy': 'Năng lượng', 'Emissions': 'Khí thải', 'Carbon emissions': 'Phát thải carbon',
  'Location': 'Địa điểm', 'Model': 'Mô hình', 'Data source': 'Nguồn dữ liệu',
  'Training time': 'Thời gian huấn luyện', 'Hardware': 'Phần cứng',
  'Sentence': 'Câu', 'Layer': 'Lớp', 'Head': 'Đầu chú ý', 'Heads': 'Các đầu chú ý',
  'Attention heads': 'Các đầu chú ý', 'Search': 'Tìm kiếm', 'Filter': 'Lọc',
  'Close': 'Đóng', 'Cancel': 'Hủy', 'Apply': 'Áp dụng', 'Submit': 'Xác nhận',
  'Next': 'Tiếp', 'Previous': 'Trước', 'Loading...': 'Đang tải…',
  'Energy Profiles': 'Hồ sơ năng lượng', 'Import': 'Nhập dữ liệu',
  'Interactive Energy Tracking for ML Models': 'Theo dõi năng lượng của mô hình học máy',
  'Your Model’s': 'Mô hình của bạn:', "Your Region's Energy Intensity": 'Cường độ phát thải tại khu vực',
  '(Lower is Better)': '(Thấp hơn là tốt hơn)', 'Enery Consumption (kWh)': 'Điện năng tiêu thụ (kWh)',
  'Select a Dataset': 'Chọn bộ dữ liệu', 'Select': 'Chọn', 'Description': 'Mô tả',
  'Instructions for adding a custom dataset': 'Hướng dẫn thêm bộ dữ liệu riêng',
  'Data Distribution': 'Phân bố dữ liệu', 'Epoch': 'Vòng huấn luyện',
  'Use pre-trained model': 'Dùng mô hình đã huấn luyện',
  'Model Overview Graph': 'Sơ đồ tổng quan mô hình', 'Layered Distributions': 'Phân bố theo lớp',
  'Gradients': 'Các gradient', 'Real': 'Thật', 'Fake': 'Giả',
  'Generated Samples': 'Mẫu được sinh', 'Training Progress': 'Tiến trình huấn luyện',
  'Learning Rate': 'Tốc độ học', 'Number of layers': 'Số lớp',
  'Number of neurons': 'Số nơ-ron', 'Activation': 'Kích hoạt',
  'Underperforming Subgroups': 'Các nhóm có hiệu suất thấp',
  'Well-performing Subgroups': 'Các nhóm có hiệu suất tốt',
  'Visual Auditor for Bias in CNN Image Classifier': 'Kiểm tra thiên lệch của bộ phân loại ảnh CNN',
  'Compare two prompts': 'So sánh hai mô tả', 'Denoising': 'Khử nhiễu',
  'Text Representation': 'Biểu diễn văn bản', 'Image Representation': 'Biểu diễn ảnh',
  'Language Model': 'Mô hình ngôn ngữ', 'Tokenize': 'Tách token',
  'Generate next token': 'Sinh token tiếp theo', 'Show all': 'Hiện tất cả',
  'Attention Head': 'Đầu chú ý', 'Attention Heads': 'Các đầu chú ý',
  'Examples': 'Ví dụ',
  'Multi-head Self Attention': 'Tự chú ý đa đầu', 'Probabilities': 'Xác suất',
  'Residual': 'Kết nối dư', 'Key': 'Khóa', 'Query': 'Truy vấn', 'Value': 'Giá trị',
  '11 more identical': '11 khối tương tự nữa', 'Blocks': 'Các khối',
  'Test your own input text': 'Nhập văn bản của bạn',
  'Try the examples while GPT-2 model is being downloaded (600MB)': 'Thử các ví dụ trong lúc mô hình GPT-2 đang được tải (600 MB)',
  'Try the examples. Please use a desktop computer to input GPT-2 prompts directly.': 'Thử các ví dụ. Dùng máy tính để nhập mô tả trực tiếp cho GPT-2.',
  'Activation Score Threshold': 'Ngưỡng điểm kích hoạt', 'by accuracy': 'theo độ chính xác',
  'Biased': 'Thiên lệch', 'Similar Well-performing Subgroup': 'Nhóm tương tự có hiệu suất tốt',
  'Confusion matrix': 'Ma trận nhầm lẫn', 'Underperforming Subgroup': 'Nhóm hiệu suất thấp',
  'True label': 'Nhãn thật', 'Predicted label': 'Nhãn dự đoán',
  'Save': 'Lưu', 'Download': 'Tải xuống', 'Edit': 'Chỉnh sửa',
  'Carbon Emissions (CO2 lbs)': 'Phát thải carbon (CO2 lb)',
  'How Your': 'Cách tính', 'Emissions are Calculated': 'khí thải của bạn',
  'Electricity (kWH)': 'Điện năng (kWh)', 'Kilowatt Hours (kWH)': 'Kilowatt giờ (kWh)',
  'Iterations per step': 'Số vòng lặp mỗi bước', 'Step': 'Bước', 'Steps': 'Các bước',
  'Slow motion': 'Chuyển động chậm', 'Reset model': 'Đặt lại mô hình',
  'Train': 'Huấn luyện', 'Start training': 'Bắt đầu huấn luyện',
  'Generated samples': 'Các mẫu được sinh', 'Discriminator Loss': 'Mất mát mạng phân biệt',
  'Generator Loss': 'Mất mát mạng sinh', 'Discriminator loss': 'Mất mát mạng phân biệt',
  'Generator loss': 'Mất mát mạng sinh', 'Real samples': 'Các mẫu thật',
  'Metric': 'Chỉ số', 'Metrics': 'Các chỉ số', 'Size': 'Kích thước',
  'Support': 'Số mẫu', 'True Positive Rate': 'Tỷ lệ dương tính đúng',
  'False Positive Rate': 'Tỷ lệ dương tính sai', 'Suggested Subgroups': 'Các nhóm con gợi ý',
  'Selected Subgroup': 'Nhóm con đã chọn', 'Feature Distribution': 'Phân bố đặc trưng',
  'Explore': 'Khám phá', 'Attention Flow': 'Luồng chú ý', 'Saliency': 'Độ nổi bật',
  'Dependency': 'Quan hệ phụ thuộc', 'Instance': 'Mẫu dữ liệu',
  'Text Representation Generator': 'Bộ tạo vector văn bản',
  'Image Representation Refiner': 'Bộ khử nhiễu ảnh',
  'Refined': 'Đã tinh chỉnh', 'Representation': 'Biểu diễn', 'Upscale': 'Phóng lớn',
  'Random noise': 'Nhiễu ngẫu nhiên', 'at timestep 0': 'tại bước 0',
  'Select another prompt': 'Chọn mô tả khác',
  'Ground Truth': 'Nhãn thật', 'Pred': 'Dự đoán',
  'Current Sentence': 'Câu hiện tại', 'Saliency View': 'Độ nổi bật',
  'Dependency List': 'Danh sách quan hệ', 'Dependency Tree': 'Cây quan hệ',
  'Show Comparison': 'Hiện so sánh', 'Syntactic Relations': 'Quan hệ cú pháp',
  'syntactic correlations': 'tương quan cú pháp', 'semantic correlations': 'tương quan ngữ nghĩa',
  'importance scores': 'điểm quan trọng', 'Attention Wights': 'Trọng số chú ý',
  'Semantic Attention Graph': 'Đồ thị chú ý ngữ nghĩa', 'Force Layout': 'Bố cục lực',
  'Radial Layout': 'Bố cục hướng tâm', 'Grid Layout': 'Bố cục lưới',
  'Edge Force': 'Lực cạnh', 'auto': 'Tự động', 'Text Order': 'Thứ tự văn bản',
  'ManyBody': 'Lực giữa các nút', 'Node Distance': 'Khoảng cách nút',
  'Edge Threshold': 'Ngưỡng cạnh', 'Semantic Saliency Score': 'Điểm nổi bật ngữ nghĩa',
  'Attention Head Overview': 'Tổng quan đầu chú ý', 'Show Detail': 'Hiện chi tiết',
  'Select Input Sentence': 'Chọn câu đầu vào', 'True': 'Nhãn thật', 'Predicted': 'Dự đoán',
  'Error': 'Sai lệch', 'Goal': 'Mục tiêu',
  'Audit Classification Models for Intersectional Bias': 'Kiểm tra thiên lệch giao thoa trong mô hình phân loại',
  'Predict income >$50K': 'Dự đoán thu nhập trên 50.000 USD',
  'Predict recidivism': 'Dự đoán tái phạm', 'Full Paper': 'Bài báo đầy đủ',
  'Code': 'Mã nguồn', 'Cite': 'Trích dẫn',
  'Epochs (': 'Số vòng (', 'extrapolated)': 'ước tính)',
  'Epoch Power Consumption': 'Điện năng theo vòng huấn luyện',
  'Total Instantaneous Power': 'Tổng công suất tức thời',
  'Your Hardware': 'Phần cứng của bạn', 'Add Alternative Hardware': 'Thêm phần cứng để so sánh',
  'Add': 'Thêm', 'Well-performing Similar Subgroup': 'Nhóm tương tự có hiệu suất tốt',
  'Smiling': 'Cười', 'Not Smiling': 'Không cười', 'Not': 'Không', 'Both': 'Cả hai',
  'Log loss': 'Mất mát log', 'LeastSq loss': 'Mất mát bình phương',
  'update(s) per epoch': 'lần cập nhật mỗi vòng', 'Optimizer': 'Bộ tối ưu',
  'Prediction of': 'Dự đoán của', 'Samples': 'Các mẫu',
  'hidden layer(s)': 'lớp ẩn', 'neuron(s)': 'nơ-ron',
  '1D Uniform': 'Đều 1D', '1D Gaussian': 'Gauss 1D',
  '2D Uniform': 'Đều 2D', '2D Gaussian': 'Gauss 2D',
  'Generator derives samples from noise': 'Generator tạo mẫu từ noise',
  'Discriminator classifies samples': 'Discriminator phân loại mẫu',
  'Computes discriminator loss/gradients': 'Tính loss và gradients của Discriminator',
  'Updates discriminator based on gradients': 'Cập nhật Discriminator theo gradients',
  'Discriminator classifies fake samples only': 'Discriminator chỉ phân loại mẫu giả',
  'Computes generator loss/gradients': 'Tính loss và gradients của Generator',
  'Updates generator based on gradients': 'Cập nhật Generator theo gradients',
  'Draw a distribution above, then click the apply button.': 'Vẽ phân bố phía trên, rồi nhấn Áp dụng.',
  'Back to Overview': 'Về tổng quan', 'Go back to Overview to expand': 'Về tổng quan để mở rộng',
  'Same random noise': 'Cùng nhiễu ngẫu nhiên', 'Evolution Trajectory': 'Quỹ đạo biến đổi',
  'Image representations projected onto 2D space': 'Biểu diễn ảnh được chiếu lên không gian 2D',
  'Tokenizer': 'Bộ tách token', '77 tokens': '77 token',
  'Image Encoder': 'Bộ mã hóa ảnh', 'Hybrid Text-Image Space': 'Không gian văn bản–ảnh',
  'Guide image generation': 'Định hướng sinh ảnh',
  'Click to see how modifying this phrase changes generated image': 'Nhấn để xem thay đổi cụm từ này ảnh hưởng đến ảnh được sinh như thế nào',
  'Dropout': 'Loại bỏ ngẫu nhiên', 'Layer Normalization': 'Chuẩn hóa lớp',
  'Dot product': 'Tích vô hướng', 'Scaling · Mask': 'Điều chỉnh · Mặt nạ',
  'Out': 'Đầu ra', 'more identical': 'khối tương tự nữa',
};

export function translateToolLabel(value: string): string {
  if (isTechnicalTerm(value)) return value;
  const trimmed = value.trim();
  const normalized = trimmed.replace(/\s+/g, ' ').replace(/:$/, '').toLowerCase();
  const key = Object.keys(labels).find(label => label.toLowerCase() === normalized);
  return key ? value.replace(trimmed, labels[key] + (trimmed.endsWith(':') ? ':' : '')) : value;
}

export function localizeTool(root: HTMLElement) {
  const translate = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE && !node.parentElement?.closest('script, style, textarea, [contenteditable], .material-icons, .cell .label, .token-id, .editable, .token-string, .softmax .text-box')) {
      const before = node.textContent ?? '';
      const after = translateToolLabel(before);
      if (before !== after) node.textContent = after;
    }
    if (node instanceof Element) {
      for (const attribute of ['title', 'aria-label', 'placeholder']) {
        const before = node.getAttribute(attribute);
        if (before !== null) {
          const after = translateToolLabel(before);
          if (before !== after) node.setAttribute(attribute, after);
        }
      }
    }
    for (const child of node.childNodes) translate(child);
  };
  translate(root);
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'childList') for (const child of record.addedNodes) translate(child);
      else translate(record.target);
    }
  });
  observer.observe(root, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['title', 'aria-label', 'placeholder'] });
  window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
}
