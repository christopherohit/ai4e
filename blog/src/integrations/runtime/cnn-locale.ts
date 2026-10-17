import { isTechnicalTerm } from './technical-terms';

export const CNN_TRANSLATIONS: Record<string, string> = {
  'Show detail': 'Hiện chi tiết',
  Unit: 'Theo lớp', Module: 'Theo khối', Global: 'Toàn mạng',
  'Change color scale range': 'Đổi phạm vi thang màu',
  'Add new input image': 'Thêm ảnh đầu vào',
  'Add Input Image': 'Thêm ảnh đầu vào',
  'Paste URL of image...': 'Dán đường dẫn ảnh…',
  Upload: 'Tải ảnh lên', or: 'hoặc', close: 'Đóng', Close: 'Đóng',
  'Play animation': 'Chạy hoặc tạm dừng mô phỏng',
  'Jump to article section': 'Đọc phần giải thích',
  'Move to article section': 'Đọc phần giải thích',
  'Hover over': 'Di chuột lên',
  'the matrices to change kernel position.': 'các ma trận để đổi vị trí kernel.',
  'the matrices to change pixel.': 'các ma trận để đổi điểm ảnh.',
  'the numbers to highlight logit circles.': 'các con số để làm nổi bật điểm logit.',
  'Input Size:': 'Kích thước đầu vào:', 'Padding:': 'Đệm:',
  'Kernel Size:': 'Kích thước bộ lọc:', 'Stride:': 'Bước trượt:',
  Red: 'Đỏ', Green: 'Lục', Blue: 'Lam', channel: 'kênh',
  input: 'đầu vào', output: 'đầu ra', conv: 'tích chập', max_pool: 'gộp cực đại',
  lifeboat: 'xuồng cứu sinh', ladybug: 'bọ rùa', pizza: 'pizza',
  'bell pepper': 'ớt chuông', 'school bus': 'xe buýt trường học',
  koala: 'gấu koala', espresso: 'cà phê espresso', 'red panda': 'gấu trúc đỏ',
  orange: 'cam', 'sports car': 'xe thể thao',
  Convolution: 'Tích chập', Pooling: 'Gộp cực đại', Activation: 'Kích hoạt',
  Kernel: 'Bộ lọc', Bias: 'Hệ số chệch', bias: 'hệ số chệch',
  intermediate: 'trung gian', weight: 'trọng số', element: 'phần tử',
  flatten: 'trải phẳng', 'class probabilities': 'xác suất các lớp',
  Normalize: 'Chuẩn hóa', into: 'thành', After: 'Sau', normalization: 'chuẩn hóa',
  'Slide kernel over input channel': 'Trượt kernel trên input channel',
  'to get intermediate result': 'để có kết quả trung gian',
  'Each input chanel': 'Mỗi kênh đầu vào',
  'gets a different kernel': 'dùng một kernel khác nhau',
  'Add up all intermediate': 'Cộng các kết quả trung gian',
  'results and then add bias': 'rồi thêm bias',
  'Add up all products': 'Cộng tất cả các tích',
  'and then': 'rồi thêm', Click: 'Nhấp', 'to learn more': 'để tìm hiểu thêm',
  'matrix to': 'ma trận để', 'see how it is flattened': 'xem cách trải phẳng',
  'Same flattening': 'Cùng phép trải phẳng', 'operation for': 'áp dụng cho',
  'each neuron': 'mỗi nơ-ron', 'to see': 'để xem', its: 'của nó', value: 'giá trị',
};

export function translateCnnText(value: string): string {
  if (isTechnicalTerm(value)) return value;
  const trimmed = value.trim();
  const translated = CNN_TRANSLATIONS[trimmed];
  if (translated) return value.replace(trimmed, translated);
  return value.replace('Output value:', 'Giá trị đầu ra:');
}

export function localizeCnn(root: HTMLElement) {
  function translateNode(node: Node) {
    if (node.nodeType === Node.TEXT_NODE && node.parentElement && !node.parentElement.closest('script, style')) {
      const before = node.textContent ?? '';
      const after = translateCnnText(before);
      if (before !== after) node.textContent = after;
    }
    if (node instanceof Element) {
      for (const attribute of ['title', 'aria-label', 'placeholder', 'alt']) {
        const before = node.getAttribute(attribute);
        if (before !== null) {
          const after = translateCnnText(before);
          if (before !== after) node.setAttribute(attribute, after);
        }
      }
    }
    for (const child of node.childNodes) translateNode(child);
  }
  translateNode(root);
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'childList') {
        for (const node of record.addedNodes) translateNode(node);
      } else translateNode(record.target);
    }
  });
  observer.observe(root, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['title', 'aria-label', 'placeholder', 'alt'] });
  window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
}
