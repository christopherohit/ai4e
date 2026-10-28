// Vietnamese texts for the MedLab body atlas: systems, regions and the organs listed in the sidebar. Individual
// structures keep their original English source names (BodyParts3D / HRA), as the attribution panel explains.
import type { RegionId, SystemId } from './atlas';

export const SYSTEM_INFO: Record<SystemId, { label: string; color: string; note: string }> = {
  skeletal: { label: 'Xương', color: '#e6dcc0', note: 'Khung nâng đỡ, bảo vệ cơ quan và là điểm bám của cơ; tủy xương tạo máu.' },
  muscular: { label: 'Cơ', color: '#c4675b', note: 'Cơ vân co kéo qua gân để cử động khớp, giữ tư thế và sinh nhiệt.' },
  cardiac: { label: 'Tim', color: '#e0645c', note: 'Bơm cơ bốn buồng; các van giữ cho máu chảy một chiều.' },
  arterial: { label: 'Động mạch', color: '#e5483f', note: 'Dẫn máu từ tim đi nuôi mô (hoặc lên phổi ở vòng tiểu tuần hoàn).' },
  venous: { label: 'Tĩnh mạch', color: '#5b8fd6', note: 'Đưa máu từ mô trở về tim.' },
  nervous: { label: 'Thần kinh', color: '#f0c46e', note: 'Não, tủy sống và dây thần kinh tiếp nhận, xử lý và truyền tín hiệu.' },
  sensory: { label: 'Giác quan', color: '#9fd3df', note: 'Cấu trúc của mắt và tai tiếp nhận ánh sáng, âm thanh, thăng bằng.' },
  respiratory: { label: 'Hô hấp', color: '#f19aa6', note: 'Đường dẫn khí và phổi, nơi O₂ và CO₂ trao đổi với máu.' },
  digestive: { label: 'Tiêu hóa', color: '#e5a66d', note: 'Ống tiêu hóa phân giải thức ăn, hấp thu chất dinh dưỡng và nước.' },
  urinary: { label: 'Tiết niệu', color: '#d98a64', note: 'Thận lọc máu; nước tiểu qua niệu quản xuống bàng quang.' },
  reproductive: { label: 'Sinh sản', color: '#e7a7b8', note: 'Cơ quan tạo giao tử, hormone sinh dục và (ở nữ) nuôi thai.' },
  endocrine: { label: 'Nội tiết', color: '#d7b0f0', note: 'Tuyến tiết hormone vào máu để điều hòa chuyển hóa, stress, tăng trưởng.' },
  lymphatic: { label: 'Bạch huyết', color: '#8fd19e', note: 'Lách, tuyến ức và mạch bạch huyết tham gia miễn dịch, dẫn lưu dịch mô.' },
  connective: { label: 'Mô liên kết', color: '#a9c9c0', note: 'Sụn, dây chằng, màng gian cốt nối và ổn định các cấu trúc.' },
};

export const DEFAULT_SYSTEMS: SystemId[] = ['skeletal', 'cardiac', 'nervous', 'sensory', 'respiratory', 'digestive', 'urinary', 'reproductive', 'endocrine', 'lymphatic', 'connective'];

export const REGION_INFO: Record<RegionId, { label: string; hint: string }> = {
  head: { label: 'Đầu và cổ', hint: 'Sọ, não, mắt, hầu họng, đốt sống cổ' },
  thorax: { label: 'Ngực', hint: 'Lồng ngực, tim, phổi, khí quản' },
  abdomen: { label: 'Bụng & vùng chậu', hint: 'Gan, dạ dày, ruột, thận, chậu hông' },
  arm: { label: 'Cánh tay & bàn tay', hint: 'Vai, cánh tay, cẳng tay, bàn tay' },
  leg: { label: 'Chân & bàn chân', hint: 'Đùi, gối, cẳng chân, bàn chân' },
};

export interface OrganText {
  name: string;
  latin: string;
  system: SystemId;
  card?: boolean;
  sex?: 'male' | 'female';
  description: string;
  function: string;
  location: string;
  fact: string;
}

export const ORGANS: Record<string, OrganText> = {
  heart: {
    name: 'Tim', latin: 'cor', system: 'cardiac', card: true,
    description: 'Tim là một khối cơ rỗng gồm hai tâm nhĩ và hai tâm thất. Nửa phải đẩy máu nghèo oxy lên phổi, nửa trái đẩy máu giàu oxy đi khắp cơ thể.',
    function: 'Co bóp nhịp nhàng để duy trì tuần hoàn; bốn van (ba lá, hai lá, động mạch phổi, động mạch chủ) giữ máu đi một chiều.',
    location: 'Trong trung thất giữa của lồng ngực, sau xương ức, mỏm tim lệch sang trái, nằm trên cơ hoành.',
    fact: 'Ở người trưởng thành lúc nghỉ, tim đập khoảng 60–100 lần mỗi phút, tức hơn 100 000 nhịp mỗi ngày.',
  },
  brain: {
    name: 'Não', latin: 'encephalon', system: 'nervous', card: true,
    description: 'Não gồm đại não, gian não, thân não và tiểu não. Vỏ đại não chia thành các hồi và thùy; bên trong là chất trắng và hệ thống não thất chứa dịch não tủy.',
    function: 'Tiếp nhận cảm giác, điều khiển vận động, ghi nhớ, ngôn ngữ; thân não điều hòa nhịp thở và nhịp tim.',
    location: 'Trong hộp sọ, được bao bởi ba lớp màng não và đệm bởi dịch não tủy.',
    fact: 'Não người trưởng thành nặng khoảng 1,3–1,4 kg (≈2% khối lượng cơ thể) nhưng dùng khoảng 20% năng lượng lúc nghỉ.',
  },
  lungs: {
    name: 'Phổi', latin: 'pulmones', system: 'respiratory', card: true,
    description: 'Phổi phải có ba thùy, phổi trái có hai thùy và chừa chỗ cho tim. Cây phế quản chia nhánh nhiều lần trước khi kết thúc ở các phế nang.',
    function: 'Trao đổi khí: oxy khuếch tán từ phế nang vào máu, CO₂ đi theo chiều ngược lại.',
    location: 'Hai bên trung thất trong lồng ngực, đáy phổi tựa lên cơ hoành, đỉnh phổi nhô lên trên xương sườn I.',
    fact: 'Hai phổi có khoảng 300–500 triệu phế nang; diện tích trao đổi khí cỡ 50–100 m².',
  },
  liver: {
    name: 'Gan', latin: 'hepar', system: 'digestive', card: true,
    description: 'Gan là tạng đặc lớn nhất trong cơ thể. Ở đây gan hiện qua các phân thùy cùng hệ tĩnh mạch gan, tĩnh mạch cửa và đường mật bên trong.',
    function: 'Chuyển hóa chất dinh dưỡng, dự trữ glycogen, tổng hợp protein huyết tương, khử độc và tiết mật.',
    location: 'Phần lớn ở hạ sườn phải, ngay dưới vòm cơ hoành, lan sang thượng vị.',
    fact: 'Gan người trưởng thành nặng khoảng 1,4–1,6 kg và có khả năng tái tạo sau khi bị cắt bớt một phần.',
  },
  kidneys: {
    name: 'Thận', latin: 'renes', system: 'urinary', card: true,
    description: 'Hai quả thận hình hạt đậu, mỗi quả chứa khoảng một triệu nephron là đơn vị lọc máu.',
    function: 'Lọc máu, tạo nước tiểu, cân bằng nước – điện giải – acid base và tiết hormone (renin, erythropoietin).',
    location: 'Sau phúc mạc, hai bên cột sống ngang đốt sống ngực XII đến thắt lưng III; thận phải thường thấp hơn thận trái.',
    fact: 'Mỗi ngày thận lọc khoảng 180 lít dịch nhưng chỉ thải ra 1–2 lít nước tiểu.',
  },
  stomach: {
    name: 'Dạ dày', latin: 'gaster', system: 'digestive', card: true,
    description: 'Đoạn phình to của ống tiêu hóa nối thực quản với tá tràng, gồm tâm vị, đáy vị, thân vị và môn vị.',
    function: 'Chứa và nhào trộn thức ăn, tiết acid HCl và pepsin để bắt đầu tiêu hóa protein.',
    location: 'Vùng thượng vị và hạ sườn trái, dưới cơ hoành.',
    fact: 'Dịch vị có pH khoảng 1,5–3,5; lớp chất nhầy giúp niêm mạc không tự tiêu hóa chính nó.',
  },
  'small-intestine': {
    name: 'Ruột non', latin: 'intestinum tenue', system: 'digestive', card: true,
    description: 'Gồm tá tràng, hỗng tràng và hồi tràng. Trong mô hình, hỗng tràng và hồi tràng được chia thành nhiều đoạn có thể chọn riêng.',
    function: 'Nơi tiêu hóa hóa học hoàn tất và hấp thu phần lớn chất dinh dưỡng qua nhung mao.',
    location: 'Chiếm phần giữa và dưới ổ bụng, được khung đại tràng bao quanh.',
    fact: 'Ruột non dài khoảng 6–7 m ở người sống, là đoạn dài nhất của ống tiêu hóa.',
  },
  'large-intestine': {
    name: 'Ruột già', latin: 'intestinum crassum', system: 'digestive', card: true,
    description: 'Gồm manh tràng, ruột thừa, đại tràng lên – ngang – xuống, đại tràng sigma và trực tràng; các dải cơ dọc (dải sán) nhìn rõ trên mô hình.',
    function: 'Hấp thu nước và điện giải, tạo và lưu trữ phân; hệ vi khuẩn đường ruột lên men chất xơ.',
    location: 'Tạo thành khung quanh ruột non trong ổ bụng, kết thúc ở trực tràng trong chậu hông.',
    fact: 'Ruột già dài khoảng 1,5 m nhưng đường kính lớn hơn ruột non.',
  },
  pancreas: {
    name: 'Tụy', latin: 'pancreas', system: 'digestive', card: true,
    description: 'Tuyến dài nằm ngang gồm đầu, thân và đuôi tụy, có ống tụy đổ vào tá tràng.',
    function: 'Ngoại tiết: enzyme tiêu hóa. Nội tiết: đảo Langerhans tiết insulin và glucagon điều hòa đường huyết.',
    location: 'Sau dạ dày, sau phúc mạc; đầu tụy nằm trong khung tá tràng, đuôi tụy hướng về lách.',
    fact: 'Đảo tụy chỉ chiếm khoảng 1–2% khối tụy nhưng quyết định lượng đường trong máu.',
  },
  spleen: {
    name: 'Lách', latin: 'splen', system: 'lymphatic', card: true,
    description: 'Cơ quan lympho lớn nhất, mềm và nhiều máu.',
    function: 'Lọc máu, loại bỏ hồng cầu già, tham gia đáp ứng miễn dịch.',
    location: 'Hạ sườn trái, sau dạ dày, ngang xương sườn IX–XI.',
    fact: 'Hồng cầu sống khoảng 120 ngày; lách là nơi chính phân hủy các hồng cầu đã già.',
  },
  bladder: {
    name: 'Bàng quang', latin: 'vesica urinaria', system: 'urinary',
    description: 'Túi cơ rỗng có thành co giãn. Phiên bản nữ dùng mô hình bàng quang riêng gồm đáy, cổ và tam giác bàng quang.',
    function: 'Chứa nước tiểu và tống xuất khi đi tiểu nhờ cơ bức niệu.',
    location: 'Trong chậu hông, sau xương mu; ở nữ nằm trước tử cung và âm đạo, ở nam nằm trên tuyến tiền liệt.',
    fact: 'Cảm giác buồn tiểu thường xuất hiện khi bàng quang chứa khoảng 200–300 ml.',
  },
  gallbladder: {
    name: 'Túi mật', latin: 'vesica biliaris', system: 'digestive',
    description: 'Túi hình quả lê áp vào mặt dưới gan, nối với ống mật chủ qua ống túi mật.',
    function: 'Dự trữ và cô đặc mật; co bóp đẩy mật xuống tá tràng sau bữa ăn nhiều chất béo.',
    location: 'Mặt dưới thùy phải của gan.',
    fact: 'Túi mật chứa khoảng 30–50 ml mật.',
  },
  esophagus: {
    name: 'Thực quản', latin: 'oesophagus', system: 'digestive',
    description: 'Ống cơ nối hầu với dạ dày.',
    function: 'Đẩy thức ăn xuống dạ dày bằng nhu động.',
    location: 'Đi trong cổ và trung thất sau, xuyên cơ hoành để vào dạ dày.',
    fact: 'Thực quản dài khoảng 25 cm.',
  },
  trachea: {
    name: 'Khí quản', latin: 'trachea', system: 'respiratory',
    description: 'Ống dẫn khí được giữ mở bởi các vòng sụn hình chữ C.',
    function: 'Dẫn khí từ thanh quản xuống hai phế quản chính.',
    location: 'Từ cổ (ngang đốt sống cổ VI) xuống trung thất, chia đôi ở ngang đốt sống ngực IV–V.',
    fact: 'Khí quản dài khoảng 10–12 cm với 16–20 vòng sụn.',
  },
  ureters: {
    name: 'Niệu quản', latin: 'ureteres', system: 'urinary',
    description: 'Hai ống cơ nhỏ dẫn nước tiểu từ bể thận xuống bàng quang.',
    function: 'Nhu động đẩy nước tiểu xuống bàng quang.',
    location: 'Sau phúc mạc, dọc hai bên cột sống thắt lưng rồi vào chậu hông.',
    fact: 'Mỗi niệu quản dài khoảng 25–30 cm.',
  },
  thymus: {
    name: 'Tuyến ức', latin: 'thymus', system: 'lymphatic',
    description: 'Cơ quan lympho trung ương gồm hai thùy.',
    function: 'Nơi tế bào lympho T trưởng thành và học cách không tấn công mô của chính cơ thể.',
    location: 'Trung thất trên, sau xương ức, trước tim và các mạch lớn.',
    fact: 'Tuyến ức lớn nhất ở tuổi dậy thì rồi dần được thay bằng mô mỡ.',
  },
  adrenals: {
    name: 'Tuyến thượng thận', latin: 'glandulae suprarenales', system: 'endocrine',
    description: 'Hai tuyến nhỏ úp trên cực trên của thận, gồm vỏ và tủy thượng thận.',
    function: 'Vỏ tiết cortisol, aldosterone; tủy tiết adrenaline và noradrenaline khi căng thẳng.',
    location: 'Sau phúc mạc, trên đỉnh hai thận.',
    fact: 'Mỗi tuyến chỉ nặng khoảng 4–5 g.',
  },
  eyes: {
    name: 'Mắt', latin: 'oculi', system: 'sensory',
    description: 'Nhãn cầu cùng các cơ ngoài nhãn cầu và bộ lệ: giác mạc, mống mắt, thể mi, thủy tinh thể, thể kính, võng mạc, củng mạc.',
    function: 'Hội tụ ánh sáng lên võng mạc, nơi tế bào cảm quang chuyển ánh sáng thành tín hiệu thần kinh.',
    location: 'Trong hai ổ mắt của hộp sọ.',
    fact: 'Nhãn cầu người trưởng thành có đường kính khoảng 24 mm.',
  },
  skull: {
    name: 'Hộp sọ', latin: 'cranium', system: 'skeletal',
    description: 'Gồm sọ não bao quanh não và sọ mặt tạo khung cho mặt, mũi, ổ mắt, hàm.',
    function: 'Bảo vệ não và giác quan, làm chỗ bám cho cơ nhai và cơ mặt.',
    location: 'Trên cùng của cột sống.',
    fact: 'Hộp sọ người trưởng thành thường được mô tả gồm 22 xương (8 xương sọ não, 14 xương sọ mặt), chưa kể xương con ở tai.',
  },
  spine: {
    name: 'Cột sống', latin: 'columna vertebralis', system: 'skeletal',
    description: 'Chuỗi đốt sống cổ, ngực, thắt lưng, xương cùng và xương cụt, có các đĩa gian đốt ở giữa.',
    function: 'Nâng đỡ thân, bảo vệ tủy sống và cho phép cúi, ngửa, xoay.',
    location: 'Đường giữa phía sau thân, từ nền sọ đến xương cụt.',
    fact: 'Thường có 33 đốt: 7 cổ, 12 ngực, 5 thắt lưng, 5 đốt cùng dính thành xương cùng và khoảng 4 đốt cụt.',
  },
  ribcage: {
    name: 'Lồng ngực', latin: 'cavea thoracis', system: 'skeletal',
    description: 'Khung xương gồm 12 đôi xương sườn, sụn sườn và xương ức, khớp với 12 đốt sống ngực.',
    function: 'Bảo vệ tim, phổi; thay đổi thể tích khi thở.',
    location: 'Phần trên của thân, từ nền cổ đến bờ sườn.',
    fact: 'Hai đôi sườn cuối (XI, XII) là “sườn cụt” vì không nối với xương ức.',
  },
  pelvis: {
    name: 'Khung chậu', latin: 'pelvis', system: 'skeletal',
    description: 'Vòng xương gồm hai xương chậu và xương cùng.',
    function: 'Truyền trọng lượng thân xuống hai chân, bảo vệ cơ quan vùng chậu.',
    location: 'Phần dưới của thân, nối cột sống với hai xương đùi.',
    fact: 'Khung chậu nữ thường rộng và nông hơn nam, phù hợp với việc sinh nở.',
  },
  diaphragm: {
    name: 'Cơ hoành', latin: 'diaphragma', system: 'muscular',
    description: 'Tấm cơ hình vòm ngăn cách khoang ngực với ổ bụng.',
    function: 'Cơ hô hấp chính: khi co, vòm hoành hạ xuống làm phổi giãn ra và hút khí vào.',
    location: 'Đáy lồng ngực, có lỗ cho thực quản, động mạch chủ và tĩnh mạch chủ dưới đi qua.',
    fact: 'Nấc cụt là do cơ hoành co thắt đột ngột không theo ý muốn.',
  },
  prostate: {
    name: 'Tuyến tiền liệt', latin: 'prostata', system: 'reproductive', sex: 'male',
    description: 'Tuyến ngoại tiết bao quanh đoạn đầu niệu đạo ở nam.',
    function: 'Tiết dịch góp phần tạo tinh dịch.',
    location: 'Ngay dưới bàng quang, trước trực tràng.',
    fact: 'Tuyến tiền liệt có kích thước cỡ một quả óc chó ở người trưởng thành.',
  },
  testes: {
    name: 'Tinh hoàn', latin: 'testes', system: 'reproductive', sex: 'male',
    description: 'Hai tuyến sinh dục nam nằm trong bìu, kèm mào tinh và ống dẫn tinh.',
    function: 'Sản xuất tinh trùng và testosterone.',
    location: 'Trong bìu, ngoài ổ bụng.',
    fact: 'Nằm ngoài ổ bụng giúp tinh hoàn mát hơn thân nhiệt vài độ, thuận lợi cho tạo tinh trùng.',
  },
  uterus: {
    name: 'Tử cung', latin: 'uterus', system: 'reproductive', sex: 'female',
    description: 'Cơ quan cơ rỗng hình quả lê gồm đáy, thân, eo và cổ tử cung.',
    function: 'Nơi phôi làm tổ và phát triển trong thai kỳ; niêm mạc bong ra theo chu kỳ kinh nguyệt.',
    location: 'Trong chậu hông, giữa bàng quang (phía trước) và trực tràng (phía sau).',
    fact: 'Ngoài thai kỳ, tử cung dài khoảng 7–8 cm.',
  },
  ovaries: {
    name: 'Buồng trứng', latin: 'ovaria', system: 'reproductive', sex: 'female',
    description: 'Hai tuyến sinh dục nữ hình hạt hạnh nhân.',
    function: 'Chứa nang trứng, phóng noãn và tiết estrogen, progesterone.',
    location: 'Hai bên tử cung, treo bởi dây chằng trên thành chậu hông.',
    fact: 'Khi sinh ra, buồng trứng có khoảng 1–2 triệu nang noãn nguyên thủy.',
  },
  'uterine-tubes': {
    name: 'Vòi tử cung', latin: 'tubae uterinae', system: 'reproductive', sex: 'female',
    description: 'Hai ống đi từ sừng tử cung tới gần buồng trứng, gồm kẽ, eo, bóng, phễu và tua vòi.',
    function: 'Đón noãn sau phóng noãn; sự thụ tinh thường xảy ra ở đoạn bóng.',
    location: 'Bờ trên dây chằng rộng, hai bên tử cung.',
    fact: 'Mỗi vòi dài khoảng 10–12 cm.',
  },
  vagina: {
    name: 'Âm đạo', latin: 'vagina', system: 'reproductive', sex: 'female',
    description: 'Ống cơ – sợi nối cổ tử cung với bên ngoài.',
    function: 'Đường ra của máu kinh, đường sinh khi sinh thường.',
    location: 'Trong chậu hông, sau bàng quang – niệu đạo, trước trực tràng.',
    fact: 'Thành âm đạo có nhiều nếp gấp giúp nó giãn rất rộng khi sinh.',
  },
  breasts: {
    name: 'Tuyến vú', latin: 'glandulae mammariae', system: 'reproductive', sex: 'female',
    description: 'Mỗi tuyến vú gồm các thùy tuyến, ống và xoang sữa đổ ra núm vú, được giữ bởi dây chằng treo.',
    function: 'Tiết sữa nuôi con sau sinh.',
    location: 'Trước cơ ngực lớn, khoảng từ xương sườn II đến VI.',
    fact: 'Mỗi tuyến có khoảng 15–20 thùy, mỗi thùy có ống dẫn sữa riêng.',
  },
};

/** Guided tour (Khám phá có hướng dẫn): organ key and the one-line narration. */
export const TOUR: [string, string][] = [
  ['skull', 'Bắt đầu từ trên: hộp sọ là chiếc mũ bảo hiểm tự nhiên của não.'],
  ['brain', 'Bên trong là não – các hồi vỏ não, chất trắng và tiểu não được dựng thành từng phần riêng.'],
  ['lungs', 'Xuống lồng ngực: hai phổi bao lấy cây phế quản; phổi phải ba thùy, phổi trái hai thùy.'],
  ['heart', 'Giữa hai phổi là tim, gồm 83 cấu trúc trong mô hình: thành tim, buồng, van và mạch vành.'],
  ['diaphragm', 'Cơ hoành ngăn ngực và bụng; mỗi lần nó co là một nhịp hít vào.'],
  ['liver', 'Ngay dưới cơ hoành bên phải là gan, tạng đặc lớn nhất cơ thể.'],
  ['stomach', 'Bên trái là dạ dày, nơi acid bắt đầu phân giải protein.'],
  ['small-intestine', 'Ruột non dài vài mét cuộn gọn giữa ổ bụng, hấp thu phần lớn dinh dưỡng.'],
  ['kidneys', 'Phía sau, hai thận lọc khoảng 180 lít dịch mỗi ngày.'],
  ['pelvis', 'Cuối cùng là khung chậu – nơi hình dáng nam và nữ khác nhau rõ nhất. Thử nút Nam/Nữ ở thanh dưới.'],
];

/** Quiz (Đánh giá): find the organ on the model. */
export const QUIZ = ['heart', 'liver', 'stomach', 'kidneys', 'spleen', 'pancreas', 'brain', 'lungs', 'large-intestine', 'bladder'];
