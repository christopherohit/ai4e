export type Scenario = 'valid' | 'invalid' | 'missing';
export const scenarios = {
  valid: { label: 'Giá trị hợp lệ', value: 42, status: 'PASS', authority: 'SUPPLIED_ECUC_DEFINITIONS', message: '42 nằm trong miền [1, 63].', facets: ['integer', 'MIN 1', 'MAX 63', 'upper-multiplicity 1'] },
  invalid: { label: 'Vượt MAX', value: 65, status: 'FAIL', authority: 'SUPPLIED_ECUC_DEFINITIONS', message: '65 lớn hơn MAX 63.', facets: ['integer', 'MIN 1', 'MAX 63', 'upper-multiplicity 1'] },
  missing: { label: 'Thiếu BSWMD', value: 42, status: 'DEGRADED', authority: 'OBSERVED_CONGRUENCE', message: 'Không có định nghĩa nhà cung cấp; chỉ quan sát cấu hình tương tự.', facets: ['observed value', 'no authoritative MIN/MAX', 'provenance uncertain'] },
} as const;
