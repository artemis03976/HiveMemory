/** 主题选项的解析：跟随系统时随系统的深浅色变化，选定的深色或浅色不受系统影响。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, resolveTheme;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ resolveTheme } = await vite.ssrLoadModule('/src/stores/ui/theme.ts'));
});
after(async () => { await vite?.close(); });

test('跟随系统时取系统的深浅色', () => {
  assert.equal(resolveTheme('system', true), 'dark');
  assert.equal(resolveTheme('system', false), 'light');
});

test('选定的深色或浅色不随系统变化', () => {
  assert.equal(resolveTheme('dark', false), 'dark');
  assert.equal(resolveTheme('light', true), 'light');
});
