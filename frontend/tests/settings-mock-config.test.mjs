/** 调试模式的模拟配置要与设置页的字段表同步：每个字段都有值，且类型与控件一致，否则调试时设置项显示为空。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, MOCK_CONFIG, COMMON_SECTIONS, ADVANCED_SECTIONS, readPath;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ MOCK_CONFIG } = await vite.ssrLoadModule('/src/constants/settings.ts'));
  ({ COMMON_SECTIONS, ADVANCED_SECTIONS, readPath } = await vite.ssrLoadModule('/src/constants/settingsSchema.ts'));
});
after(async () => { await vite?.close(); });

const EXPECTED_TYPE = { switch: 'boolean', number: 'number', select: 'string', model: 'string' };

test('模拟配置为设置页的每个字段给出与控件相符的值', () => {
  const fields = [...COMMON_SECTIONS, ...ADVANCED_SECTIONS].flatMap((section) => section.fields);
  const mismatched = fields.filter((field) => {
    const value = readPath(MOCK_CONFIG, field.path);
    if (value === undefined) return true;
    if (value === null) return !field.nullable;
    const expected = EXPECTED_TYPE[field.kind];
    return expected !== undefined && typeof value !== expected;
  });
  assert.deepEqual(mismatched.map((field) => field.path), []);
});
