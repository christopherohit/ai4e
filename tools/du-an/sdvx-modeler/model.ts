// SDVX Modeler lab data. Stage names, refusal codes and every measured figure come from the SDVX Modeler source and
// design docs (paths in the article). The proposed changes are illustrative: each is a typical agent proposal that the
// real pipeline is built to stop, replayed here against the documented stage order. No model or project is run.

/** The validation stages a candidate transaction passes before one journalled commit (workspace/validation.py). */
export const STAGES = [
  { id: 'WORKSPACE_ADMISSIBILITY', group: 'Nền', label: 'Workspace hợp lệ', what: 'Revision đã biết, không có bản sao workspace lồng nhau, file đích được phép ghi theo vai trò.' },
  { id: 'STRICT_ROUNDTRIP', group: 'Nền', label: 'Round-trip nguyên vẹn', what: 'Đọc rồi ghi lại không đổi byte nào ngoài vùng sửa.' },
  { id: 'XSD_SCHEMA', group: 'Cú pháp', label: 'XSD schema', what: 'Đúng revision AUTOSAR khai báo trong xsi:schemaLocation (7 revision được cache).' },
  { id: 'REFERENCE_INTEGRITY', group: 'Cú pháp', label: 'Tham chiếu', what: 'Mọi REF giải được, đúng meta-class đích, không mơ hồ giữa các AR path trùng.' },
  { id: 'ECUC_CONSTRAINT', group: 'Định nghĩa', label: 'Ràng buộc BSWMD', what: 'Kiểu, MIN/MAX, literal, multiplicity của từng tham số theo định nghĩa module.' },
  { id: 'ECUC_CONFORMANCE', group: 'Định nghĩa', label: 'Conformance', what: 'Cấu trúc container, choice, DESTINATION-REF; không bao giờ bị bỏ qua.' },
  { id: 'DOMAIN_INVARIANT', group: 'Miền', label: 'Invariant khai thác', what: 'Quy luật học từ baseline: định danh duy nhất, cấp số cộng, tham chiếu phủ…' },
  { id: 'SCOPED_BASELINE', group: 'Miền', label: 'So với baseline', what: 'Chỉ lỗi mới phát sinh mới bị chặn; lỗi có sẵn trong baseline không chặn việc khác.' },
  { id: 'DUPLICATE_IDENTIFIER', group: 'Miền', label: 'Định danh trùng', what: 'DID, DTC, NvM block id… không trùng với cái đã có.' },
  { id: 'EFFECT_DIFF', group: 'Hiệu ứng', label: 'Effect contract', what: 'Diff ngữ nghĩa phải khớp hiệu ứng đã khai báo: expected, optional, forbidden.' },
  { id: 'CLOSURE_ACCOUNTING', group: 'Hiệu ứng', label: 'Closure đầy đủ', what: 'Mọi vai trò bắt buộc của closure (Dcm, NvM, Rte, SWC…) đều có mặt.' },
  { id: 'TEXTUAL_CHURN', group: 'Hiệu ứng', label: 'Churn văn bản', what: 'Số byte đổi không vượt mức khai báo; 6 trường dialect của file giữ nguyên.' },
  { id: 'EXTERNAL_TOOLCHAIN', group: 'Ngoài', label: 'Toolchain ngoài', what: 'Kiểm tra bằng công cụ của nhà cung cấp khi có; nếu không có thì báo SKIPPED, không giả vờ PASS.' },
] as const;
export type StageId = typeof STAGES[number]['id'];
export type Outcome = 'PASS' | 'FAIL' | 'DEGRADED' | 'SKIPPED' | 'NOT_RUN';

export interface Proposal {
  id: string;
  title: string;
  call: string;
  /** Stage that stops it; absent = commits. */
  stopsAt?: StageId;
  degraded?: StageId[];
  code: string;
  detail: string;
  repair?: string;
  lesson: string;
}

export const PROPOSALS: Proposal[] = [
  {
    id: 'ok', title: 'Thêm DID từ một DID mẫu đã xác nhận',
    call: 'add_did_transaction(did=0xF1A0, analog=DID_VEHICLE_ID, storage="NvM")',
    code: 'COMMITTED', detail: 'changed_files: Dcm.arxml, NvM.arxml, Rte.arxml, Dcm_swc.arxml. Diff ngữ nghĩa khớp effect contract; 0 lỗi mới so với baseline.',
    degraded: ['EXTERNAL_TOOLCHAIN'], lesson: 'Một thay đổi hợp lệ đi qua mọi tầng rồi được ghi đúng một lần qua journal. Tầng toolchain ngoài báo SKIPPED vì không có DaVinci headless; hệ thống nói rõ điều đó thay vì báo PASS.',
  },
  {
    id: 'raw', title: 'Sửa thẳng văn bản file .arxml',
    call: 'replace_in_file("Config/ECUC/Dcm.arxml", "…Size>8<", "…Size>16<")',
    stopsAt: 'WORKSPACE_ADMISSIBILITY', code: 'SEMANTIC_TRANSACTION_REQUIRED',
    detail: 'Tool ghi văn bản thường không được tạo hay sửa ARXML. Nothing was written.',
    repair: 'Gợi ý: configure_dcm_* transaction, hoặc preflight_diag_change để xem trước.',
    lesson: 'Bị chặn trước mọi kiểm tra khác: quyền ghi ARXML chỉ thuộc về transaction. Lời từ chối kèm hướng đi để agent không thử lại cùng một lời gọi.',
  },
  {
    id: 'generated', title: 'Ghi vào file do công cụ sinh ra',
    call: 'configure_ecuc_container(file="DaVinci_Project/…/Dcm_Generated.arxml", …)',
    stopsAt: 'WORKSPACE_ADMISSIBILITY', code: 'PROTECTED_RESOURCE',
    detail: 'Vai trò file GENERATED_OUTPUT (nhận diện theo nội dung, độ tin cậy 1.0) có writability READ_ONLY. Bằng chứng: AUTOSAR_ROLE_WRITABILITY, “no write authority owns it”.',
    lesson: 'Quyền ghi do vai trò file quyết định (ADR-029), không do đường dẫn. Mọi file sinh tự động, BSWMD và định nghĩa ECUC đều chỉ đọc.',
  },
  {
    id: 'max', title: 'Đặt giá trị vượt MAX trong BSWMD',
    call: 'configure_ecuc_container(action="set", definition=…/DcmDspSecurityLevel, value=0x41)',
    stopsAt: 'ECUC_CONSTRAINT', code: 'INVALID_VALUE',
    detail: 'BSWMD: ECUC-INTEGER-PARAM-DEF, MIN 1, MAX 63. Giá trị 65 vượt MAX 63.',
    repair: 'Nếu agent gọi action="add" lên tham số đã có (UPPER-MULTIPLICITY 1), lời từ chối ECUC_EDIT_LEAF_ALREADY_PRESENT kèm corrected_call đổi sang "set", không tự áp dụng.',
    lesson: 'Ràng buộc đến từ file BSWMD của nhà cung cấp, không từ trí nhớ của model. Không có BSWMD thì tầng này báo DEGRADED, không bao giờ PASS.',
  },
  {
    id: 'dest', title: 'Tham chiếu sai loại đích',
    call: 'set_reference(…/DcmDspDataBlockIdRef → /Dem/DemConfigSet/DemEventParameter_X)',
    stopsAt: 'REFERENCE_INTEGRITY', code: 'REFERENCE_DESTINATION_MISMATCH',
    detail: 'DESTINATION-REF yêu cầu NvMBlockDescriptor; đích là DemEventParameter. Trạng thái giải: RESOLVED_WRONG_TARGET_TYPE.',
    lesson: 'File vẫn là XML hợp lệ và qua XSD. Chỉ bộ giải tham chiếu biết meta-class mới bắt được lỗi này.',
  },
  {
    id: 'unique', title: 'Cấp trùng NvM block id',
    call: 'add_storage_pair_transaction(block="DID_F1A0", block_id=7)',
    stopsAt: 'DOMAIN_INVARIANT', code: 'ECUC_OBSERVED_UNIQUE',
    detail: 'Invariant UNIQUE trên NvMNvramBlockIdentifier: 18 lần xuất hiện trong baseline, 0 phản ví dụ, kiểu INTEGER. Id 7 đã thuộc NvMBlock_Calibration.',
    repair: 'identity_allocation.next_in_observed_progression đề xuất id tiếp theo trong dãy quan sát được.',
    lesson: 'Invariant được khai thác từ baseline, không từ chính thay đổi, nên một transaction không thể tự hợp thức hóa lỗi của mình. Chỉ chặn khi đủ bằng chứng: ít nhất 3 lần ủng hộ, 0 phản ví dụ, có định nghĩa kiểu.',
  },
  {
    id: 'closure', title: 'Thêm DID nhưng quên phía Rte',
    call: 'add_did_transaction(…) // plan bỏ sót SWC-PORT-MAPPING',
    stopsAt: 'CLOSURE_ACCOUNTING', code: 'CLOSURE_INCOMPLETE',
    detail: 'required_closure_roles: DCM_DATA, DCM_DID, NVM_BLOCK, RTE_MAPPING, SWC_PORT. Thiếu RTE_MAPPING.',
    lesson: 'Một port bị thiếu là một sự vắng mặt, không phải một tham chiếu hỏng, nên kiểm tra tham chiếu không thấy. Kế toán closure thì thấy.',
  },
  {
    id: 'churn', title: 'Ghi lại file với kết thúc dòng LF',
    call: 'transaction commit trên file CRLF xuất từ công cụ Windows',
    stopsAt: 'TEXTUAL_CHURN', code: 'DIALECT_CHANGED',
    detail: 'Dialect của file: không BOM, indent 2, CRLF, xmlns đứng đầu, UUID chữ thường. Ứng viên đổi EOL thành LF: 3.642 dòng đổi cho một sửa 6 dòng.',
    lesson: 'Sự cố thật đã xảy ra trước khi tầng này tồn tại. Giờ mọi thay đổi về BOM, khai báo, EOL, indent, thứ tự thuộc tính gốc hay chữ hoa UUID đều bị chặn.',
  },
];

/** Run a proposal through the stages: PASS until the stopping stage, then NOT_RUN. */
export function run(proposal: Proposal): Record<StageId, Outcome> {
  const out = {} as Record<StageId, Outcome>;
  let stopped = false;
  for (const stage of STAGES) {
    if (stopped) out[stage.id] = 'NOT_RUN';
    else if (stage.id === proposal.stopsAt) { out[stage.id] = 'FAIL'; stopped = true; }
    else out[stage.id] = proposal.degraded?.includes(stage.id) ? (stage.id === 'EXTERNAL_TOOLCHAIN' ? 'SKIPPED' : 'DEGRADED') : 'PASS';
  }
  return out;
}

/** Measured figures (sources in the article). */
export const DATA = {
  // docs/arxml-agent-architecture/docs/20_reality_binding_microsar.md §20.4 (file roles), §20.8 (dialects)
  fileRoles: [['ECUC_ACTIVE', 37], ['BSW_INTERNAL_BEHAVIOR', 31], ['DEVELOPER_WORKSPACE', 15], ['SWC_GENERATED', 9], ['SYSTEM_IMPORT', 5], ['ECUC_PARALLEL', 2], ['GENERATED_OUTPUT', 1], ['TOOL_BACKUP', 1]] as [string, number][],
  dialects: [83, 19, 2, 1, 1, 1],
  definitionRoots: [['/MICROSAR', 22422], ['/bplus', 777], ['/AUTOSAR/EcucDefs', 34]] as [string, number][],
  // benchmarks/dcm-did/eval40/table.txt — 40 DID requests, same baseline, one at a time
  eval40: { legacy: { pass: 25, median: 162 }, dsh: { pass: 32, median: 67.5 }, total: 40 },
  // docs/arxml-agent-architecture/docs/26_production_scale_qualification.md
  scale: [[1, 0.092], [10, 0.808], [25, 3.77], [52, 16.505]] as [number, number][],
  // Telemetry snapshot of the in-house engine, 2026-06-22 → 2026-07-16 (git history of README.md)
  latency: [['p50', 54.96], ['p90', 709.49], ['p95', 1581.91], ['max', 3231.76]] as [string, number][],
  tokens: { prompt: 67_069_570, completion: 3_841_022 },
  time: { llm: 87.5, tools: 0.53 },
};
