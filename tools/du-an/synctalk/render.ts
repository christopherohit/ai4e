import { talkingView } from '../../_shared/radiance/talking-view';
import type { Explanations, LabView, Root, ViewContext } from '../../_shared/radiance/lab-core';

// The baked talking-head view (real test clip, AU/RMS signals, model outputs) is shared with the other talking-head
// project; what differs per project is the explanation of each step and the data folder chosen by mode.
export const explanations: Explanations = [
    ['Audio & AU', 'Audio được mã hóa bằng Audio-Visual Encoder; AU45 (chớp mắt) từ OpenFace điều khiển vùng mắt.', 'f_a = AVE(audio), e = AU45'],
    ['Tri-plane field', 'Mỗi điểm 3D tra cứu ba mặt phẳng hash grid, rồi kết hợp với audio để ra σ và màu.', 'F(x, f_a, e) → (σ, c)'],
    ['Ray sampling', 'Ray đi qua vùng đầu; chỉ các sample trên bề mặt khuôn mặt có trọng số đáng kể.', 'αᵢ = 1 − e^(−σᵢδᵢ)'],
    ['Portrait rendering', 'Volume rendering cho ra khuôn mặt; SyncTalk ghép đầu với torso và background để thành frame.', 'C = Σ Tᵢ·αᵢ·cᵢ'],
];

export function createView(root: Root, context: ViewContext): Promise<LabView> { return talkingView(root, context); }
