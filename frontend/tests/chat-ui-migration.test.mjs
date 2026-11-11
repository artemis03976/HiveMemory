/** chat-ui-store 的持久化迁移：已移除的终端标签不会在升级后复活，其他标签保持原样。 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

let vite, migrateChatUiState;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom' });
  ({ migrateChatUiState } = await vite.ssrLoadModule('/src/stores/ui/chatUiMigration.ts'));
});
after(async () => { await vite?.close(); });

test('持久化的终端标签迁移为引用记忆，记忆任务标签不变', () => {
  assert.equal(migrateChatUiState({ workspaceVisionTab: 'terminal' }, 4).workspaceVisionTab, 'context');
  assert.equal(migrateChatUiState({ workspaceVisionTab: 'memory-runtime' }, 4).workspaceVisionTab, 'memory-runtime');
});

test('原来的通用设置与各子系统分类并入常用，管理类分类保持原样', () => {
  assert.equal(migrateChatUiState({ settingsActiveCategory: 'engines' }, 5).settingsActiveCategory, 'common');
  assert.equal(migrateChatUiState({ settingsActiveCategory: 'general' }, 5).settingsActiveCategory, 'common');
  assert.equal(migrateChatUiState({ settingsActiveCategory: 'access' }, 5).settingsActiveCategory, 'access');
});
