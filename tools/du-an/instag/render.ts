import { talkingView } from '../../_shared/radiance/talking-view';
import type { Explanations, LabView, Root, ViewContext } from '../../_shared/radiance/lab-core';

// The baked talking-head view (real test clip, AU/RMS signals, model outputs) is shared with the other talking-head
// project; what differs per project is the explanation of each step and the data folder chosen by mode.
export const explanations: Explanations = [
    ['Gaussian canonical', '10,546 Gaussian khuôn mặt và 858 Gaussian miệng ở tư thế trung tính, học từ dữ liệu Obama.', 'θ = {μ, s, q, α, SH₁}'],
    ['Motion field', 'Motion network nhận audio và AU, trả về dịch chuyển Δμ cho từng Gaussian ở mỗi frame.', 'Δμ = M(μ, f_a, AU)'],
    ['Face + mouth', 'Khuôn mặt và khoang miệng render riêng, mỗi nhánh có alpha của nó.', 'I_face, A_face; I_mouth, A_mouth'],
    ['Fused rendering', 'Ghép hai nhánh: miệng nằm dưới, mặt phủ lên trên theo alpha.', 'I = I_face + I_mouth·(1 − A_face)'],
];

export function createView(root: Root, context: ViewContext): Promise<LabView> { return talkingView(root, context); }
