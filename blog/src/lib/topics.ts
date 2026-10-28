import type { CategorySlug } from './site';

export const TOPIC_SLUGS = ['nen-tang', 'thi-giac-may-tinh', 'xu-ly-ngon-ngu', 'mo-hinh-sinh', 'giai-thich-mo-hinh', 'danh-gia-thien-lech', 'nang-luong-ai', 'nhan-vat-so', 'agent-ky-thuat', 'medlab'] as const;
export type TopicSlug = typeof TOPIC_SLUGS[number];

export const TOPICS: Record<TopicSlug, { label: string; category: CategorySlug; description: string }> = {
  'nhan-vat-so': { label: 'Nhân vật số', category: 'du-an', description: 'Talking head, tái dựng 3D và điều khiển chuyển động từ tín hiệu.' },
  'nen-tang': { label: 'Nền tảng', category: 'kien-thuc', description: 'Các khái niệm và phương pháp cơ bản của AI.' },
  'thi-giac-may-tinh': { label: 'Thị giác máy tính', category: 'kien-thuc', description: 'Cách mô hình nhận biết và xử lý hình ảnh.' },
  'xu-ly-ngon-ngu': { label: 'Xử lý ngôn ngữ', category: 'kien-thuc', description: 'Token, Attention và các mô hình ngôn ngữ.' },
  'mo-hinh-sinh': { label: 'Mô hình sinh', category: 'kien-thuc', description: 'GAN, Diffusion và quá trình tạo dữ liệu mới.' },
  'giai-thich-mo-hinh': { label: 'Giải thích mô hình', category: 'du-an', description: 'Phân tích hành vi và cơ chế bên trong mô hình.' },
  'danh-gia-thien-lech': { label: 'Đánh giá thiên lệch', category: 'du-an', description: 'Kiểm tra sai lệch và hiệu suất giữa các nhóm dữ liệu.' },
  'nang-luong-ai': { label: 'Năng lượng AI', category: 'du-an', description: 'Theo dõi điện năng và phát thải của hệ thống AI.' },
  'agent-ky-thuat': { label: 'Agent kỹ thuật', category: 'du-an', description: 'Agent AI sửa file kỹ thuật một cách kiểm soát được.' },
  medlab: { label: 'MedLab', category: 'du-an', description: 'Kiến thức sinh học và y học qua mô hình 3D tương tác.' },
};

export function categoryTopics(category: CategorySlug) {
  return TOPIC_SLUGS.filter(slug => TOPICS[slug].category === category).map(slug => ({ slug, ...TOPICS[slug] }));
}
